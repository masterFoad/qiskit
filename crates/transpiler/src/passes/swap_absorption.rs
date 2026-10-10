// This code is part of Qiskit.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Move two-qubit interactions next to the SWAPs present after routing on the same qubit pair.
//!
//! A *unit* is a standard gate, a diagonal two-qubit unitary, or a `cx(a, b); D(b); cx(a, b)`
//! gadget with `D` diagonal, together with the single-qubit Pauli it commutes with on each of
//! its qubits.  Two units that commute with the same Pauli on every qubit they share commute
//! with each other, which is the only commutation rule the pass uses.  For each SWAP that is not
//! already next to a unit of its pair, the nearest same-pair unit on either side that commutes
//! with everything in between is moved right next to the SWAP; where that fails, the
//! single-qubit gates next to the SWAP may first be relabelled across it
//! (`SWAP (g x I) = (I x g) SWAP`).  A unit that sits in a same-pair run is either moved with
//! the whole run (CZ-only runs) or left alone, so that no existing block-synthesis opportunity
//! is split.  See the Python wrapper, `qiskit.transpiler.preset_passmanagers._swap_absorption`,
//! for the tail rules that leave a SWAP alone.

use hashbrown::HashMap;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::{SmallVec, smallvec};

use qiskit_circuit::Qubit;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGError, PyDAGCircuit};
use qiskit_circuit::operations::{Operation, OperationRef, StandardGate, UnitaryGate};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};

/// Sentinel for "no unit" / "no destination" in the index arrays.
const NONE: u32 = u32::MAX;

/// The single-qubit Pauli an operation commutes with on one of its qubits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pauli {
    X,
    Y,
    Z,
    /// No Pauli is known to commute with the operation on this qubit; nothing moves across it.
    Unknown,
}

impl Pauli {
    fn as_char(self) -> char {
        match self {
            Pauli::X => 'X',
            Pauli::Y => 'Y',
            Pauli::Z => 'Z',
            Pauli::Unknown => '-',
        }
    }
}

/// The Pauli a standard gate commutes with on each of its qubits, or `None` for a gate that is
/// not in the table (every qubit of it then blocks).
fn standard_gate_paulis(gate: StandardGate) -> Option<&'static [Pauli]> {
    use Pauli::{X, Y, Z};
    Some(match gate {
        StandardGate::RZ
        | StandardGate::Z
        | StandardGate::S
        | StandardGate::Sdg
        | StandardGate::T
        | StandardGate::Tdg
        | StandardGate::Phase
        | StandardGate::U1 => &[Z],
        StandardGate::RX | StandardGate::X | StandardGate::SX | StandardGate::SXdg => &[X],
        StandardGate::RY | StandardGate::Y => &[Y],
        StandardGate::CZ
        | StandardGate::CPhase
        | StandardGate::CRZ
        | StandardGate::CU1
        | StandardGate::CS
        | StandardGate::CSdg
        | StandardGate::RZZ => &[Z, Z],
        StandardGate::CX | StandardGate::CRX | StandardGate::CSX | StandardGate::RZX => &[Z, X],
        StandardGate::RXX => &[X, X],
        StandardGate::RYY => &[Y, Y],
        _ => return None,
    })
}

/// Whether every off-diagonal entry of the unitary's matrix is exactly zero (and every diagonal
/// entry finite).
fn is_diagonal(unitary: &UnitaryGate) -> bool {
    unitary
        .matrix_view()
        .indexed_iter()
        .all(|((row, col), value)| {
            if row == col {
                value.re.is_finite() && value.im.is_finite()
            } else {
                value.re == 0. && value.im == 0.
            }
        })
}

/// The Pauli an instruction commutes with on each of its qubits: `None` when it blocks on all of
/// them.  Only standard gates and diagonal two-qubit [UnitaryGate]s are recognised, so a custom
/// gate that reuses a standard name is never moved or crossed.
fn wire_paulis(inst: &PackedInstruction) -> Option<&'static [Pauli]> {
    match inst.op.view() {
        OperationRef::StandardGate(gate) => standard_gate_paulis(gate),
        OperationRef::Unitary(unitary) if unitary.num_qubits() == 2 && is_diagonal(unitary) => {
            Some(&[Pauli::Z, Pauli::Z])
        }
        _ => None,
    }
}

/// Whether the operation is a gate in the sense of Python space's `Gate` class.
fn is_gate(op: &PackedOperation) -> bool {
    op.is_gate()
        || matches!(
            op.view(),
            OperationRef::Unitary(_) | OperationRef::PauliProductRotation(_)
        )
}

/// Per-node facts about the operation, so that the search never touches the DAG again.
#[derive(Clone, Copy)]
struct NodeFlags {
    gate: bool,
    cx: bool,
    cz: bool,
    swap: bool,
    directive: bool,
}

impl NodeFlags {
    fn new(inst: &PackedInstruction) -> Self {
        let standard = inst.op.try_standard_gate();
        Self {
            gate: is_gate(&inst.op),
            cx: standard == Some(StandardGate::CX),
            cz: standard == Some(StandardGate::CZ),
            swap: standard == Some(StandardGate::Swap),
            directive: inst.op.directive(),
        }
    }
}

/// One qubit of an operation: the qubit, the operation's position on that qubit's wire, and the
/// Pauli the operation commutes with on it.
#[derive(Clone, Copy)]
struct NodeWire {
    qubit: u32,
    position: u32,
    pauli: Pauli,
}

/// One qubit of a unit: the qubit, the Pauli the unit commutes with on it, and the unit's
/// position in that qubit's sequence of units.
#[derive(Clone, Copy)]
struct UnitWire {
    qubit: u32,
    pauli: Pauli,
    position: u32,
}

/// A group of operations that moves as one.
struct Unit {
    /// Positions in topological order of the operations, in the order they are re-applied.
    members: SmallVec<[u32; 3]>,
    /// Range of the unit's qubits in [MoveSearch::unit_wires].
    wires: (u32, u32),
    /// For a compound unit (a whole CZ block), the units it was made from; otherwise empty.
    sources: Vec<u32>,
}

/// A unit to move next to a SWAP, on the side `step` (`1` after, `-1` before), after the single
/// qubit gates `hops` have been relabelled across the SWAP.
#[derive(Clone, Debug)]
pub struct Move {
    unit: u32,
    swap: u32,
    step: i64,
    hops: SmallVec<[u32; 4]>,
}

/// The plan of a run: the operations in topological order, the units, and the moves.
pub struct Plan {
    order: Vec<NodeIndex>,
    units: Vec<Unit>,
    unit_wires: Vec<UnitWire>,
    moves: Vec<Move>,
}

/// Python's `seq[low + 1:high]` for the non-negative indices used here.
fn between(seq: &[u32], low: i64, high: i64) -> &[u32] {
    let len = seq.len() as i64;
    let start = (low + 1).clamp(0, len);
    let end = high.clamp(start, len);
    &seq[start as usize..end as usize]
}

/// Working state of the move search.  Operations and qubits are referred to by index: an
/// operation by its position in the topological order, a qubit by its index in the DAG.
struct MoveSearch {
    num_nodes: usize,
    flags: Vec<NodeFlags>,
    /// `node_wires[node_wire_start[node]..node_wire_start[node + 1]]` are the node's qubits.
    node_wire_start: Vec<u32>,
    node_wires: Vec<NodeWire>,
    /// The operations on each qubit's wire, in order.
    wires: Vec<Vec<u32>>,
    /// Position on each wire of its last multi-qubit operation (`-1` for none).
    last_multi: Vec<i64>,
    /// Position on each wire of its last measurement, for the short-T2 qubits (`-1` for none).
    last_measure: Vec<i64>,
    short_t2: Vec<bool>,
    unit_of: Vec<u32>,
    units: Vec<Unit>,
    unit_wires: Vec<UnitWire>,
    /// The units on each qubit's wire, in order, with consecutive repeats collapsed.
    sequence: Vec<Vec<u32>>,
    /// Units that must not be moved (already next to a SWAP of their pair, or moved already).
    taken: Vec<bool>,
    /// The SWAP each moved unit goes to.
    destinations: Vec<u32>,
    /// Operations relabelled across a SWAP by a recorded move.
    hopped: Vec<bool>,
    /// Scratch: membership in the CZ block under construction, and the block itself.
    in_block: Vec<bool>,
    block: Vec<u32>,
    /// The SWAP pair of the current search and the hop runs on each of its qubits.
    pair: [u32; 2],
    runs: [Vec<u32>; 2],
    moves: Vec<Move>,
}

impl MoveSearch {
    fn new(dag: &DAGCircuit, order: &[NodeIndex], short_t2: &[u32]) -> Self {
        let num_qubits = dag.num_qubits();
        let num_nodes = order.len();
        let mut short = vec![false; num_qubits];
        for &qubit in short_t2 {
            if (qubit as usize) < num_qubits {
                short[qubit as usize] = true;
            }
        }
        let mut flags = Vec::with_capacity(num_nodes);
        let mut node_wire_start = Vec::with_capacity(num_nodes + 1);
        let mut node_wires = Vec::with_capacity(2 * num_nodes);
        let mut wires: Vec<Vec<u32>> = vec![Vec::new(); num_qubits];
        let mut last_multi = vec![-1; num_qubits];
        let mut last_measure = vec![-1; num_qubits];
        for (node, &index) in order.iter().enumerate() {
            let inst = dag[index].unwrap_operation();
            let qargs = dag.get_qargs(inst.qubits);
            let paulis = wire_paulis(inst);
            let measure = inst.op.name() == "measure";
            flags.push(NodeFlags::new(inst));
            node_wire_start.push(node_wires.len() as u32);
            for (i, qubit) in qargs.iter().enumerate() {
                let wire = &mut wires[qubit.index()];
                let position = wire.len() as u32;
                node_wires.push(NodeWire {
                    qubit: qubit.index() as u32,
                    position,
                    pauli: paulis.map_or(Pauli::Unknown, |paulis| paulis[i]),
                });
                wire.push(node as u32);
                if qargs.len() > 1 {
                    last_multi[qubit.index()] = position as i64;
                }
                if measure && short[qubit.index()] {
                    last_measure[qubit.index()] = position as i64;
                }
            }
        }
        node_wire_start.push(node_wires.len() as u32);
        let mut search = Self {
            num_nodes,
            flags,
            node_wire_start,
            node_wires,
            wires,
            last_multi,
            last_measure,
            short_t2: short,
            unit_of: vec![NONE; num_nodes],
            units: Vec::with_capacity(num_nodes),
            unit_wires: Vec::with_capacity(num_nodes),
            sequence: Vec::with_capacity(num_qubits),
            taken: Vec::new(),
            destinations: Vec::new(),
            hopped: vec![false; num_nodes],
            in_block: Vec::new(),
            block: Vec::new(),
            pair: [0, 0],
            runs: [Vec::new(), Vec::new()],
            moves: Vec::new(),
        };
        search.build_units();
        search.build_sequences();
        search
    }

    /// Group the operations into units.
    fn build_units(&mut self) {
        for node in 0..self.num_nodes as u32 {
            if self.unit_of[node as usize] != NONE {
                continue;
            }
            let mut members: SmallVec<[u32; 3]> = smallvec![node];
            let mut gadget = false;
            // `cx(a, b); D(b); cx(a, b)` with `D` a standard gate diagonal in the Z basis.
            if self.flags[node as usize].cx
                && let Some(middle) = self.after(node, 1)
                && self.qargs(middle).len() == 1
                && self.qargs(middle)[0].pauli == Pauli::Z
                && let Some(last) = self.after(middle, 0)
                && self.flags[last as usize].cx
                && self.same_qargs(last, node)
                && self.after(node, 0) == Some(last)
            {
                members = smallvec![node, middle, last];
                gadget = true;
            }
            let unit = self.units.len() as u32;
            for &member in &members {
                self.unit_of[member as usize] = unit;
            }
            let start = self.unit_wires.len() as u32;
            let (first, end) = (
                self.node_wire_start[node as usize] as usize,
                self.node_wire_start[node as usize + 1] as usize,
            );
            for &wire in &self.node_wires[first..end] {
                self.unit_wires.push(UnitWire {
                    qubit: wire.qubit,
                    pauli: if gadget { Pauli::Z } else { wire.pauli },
                    position: 0,
                });
            }
            let len = (end - first) as u32;
            self.units.push(Unit {
                members,
                wires: (start, len),
                sources: Vec::new(),
            });
        }
        self.taken = vec![false; self.units.len()];
        self.destinations = vec![NONE; self.units.len()];
        self.in_block = vec![false; self.units.len()];
    }

    /// Build each qubit's sequence of units and record the units' positions in them.
    fn build_sequences(&mut self) {
        self.sequence = self
            .wires
            .iter()
            .map(|wire| Vec::with_capacity(wire.len()))
            .collect();
        // A unit's first operation on a wire is its first member, whose `i`-th qubit is the
        // unit's `i`-th wire.
        for node in 0..self.num_nodes {
            let unit = self.unit_of[node];
            let (start, end) = (
                self.node_wire_start[node] as usize,
                self.node_wire_start[node + 1] as usize,
            );
            for (i, wire) in self.node_wires[start..end].iter().enumerate() {
                let seq = &mut self.sequence[wire.qubit as usize];
                if seq.last() != Some(&unit) {
                    seq.push(unit);
                    self.unit_wires[self.units[unit as usize].wires.0 as usize + i].position =
                        (seq.len() - 1) as u32;
                }
            }
        }
    }

    fn qargs(&self, node: u32) -> &[NodeWire] {
        let start = self.node_wire_start[node as usize] as usize;
        let end = self.node_wire_start[node as usize + 1] as usize;
        &self.node_wires[start..end]
    }

    fn same_qargs(&self, a: u32, b: u32) -> bool {
        let (a, b) = (self.qargs(a), self.qargs(b));
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.qubit == y.qubit)
    }

    /// The operation after `node` on the wire of its `i`-th qubit.
    fn after(&self, node: u32, i: usize) -> Option<u32> {
        let wire = self.qargs(node)[i];
        self.wires[wire.qubit as usize]
            .get(wire.position as usize + 1)
            .copied()
    }

    fn unit_wires(&self, unit: u32) -> &[UnitWire] {
        let (start, len) = self.units[unit as usize].wires;
        &self.unit_wires[start as usize..(start + len) as usize]
    }

    fn unit_wire(&self, unit: u32, qubit: u32) -> &UnitWire {
        self.unit_wires(unit)
            .iter()
            .find(|wire| wire.qubit == qubit)
            .expect("the unit acts on the qubit")
    }

    /// The unit's position in the qubit's sequence.
    fn position(&self, qubit: u32, unit: u32) -> i64 {
        self.unit_wire(unit, qubit).position as i64
    }

    fn pauli(&self, unit: u32, qubit: u32) -> Pauli {
        self.unit_wire(unit, qubit).pauli
    }

    fn num_qubits_of(&self, unit: u32) -> usize {
        self.units[unit as usize].wires.1 as usize
    }

    /// Whether the unit acts on exactly the two qubits of `pair`.
    fn on_pair(&self, unit: u32, pair: [u32; 2]) -> bool {
        let wires = self.unit_wires(unit);
        wires.len() == 2
            && ((wires[0].qubit == pair[0] && wires[1].qubit == pair[1])
                || (wires[0].qubit == pair[1] && wires[1].qubit == pair[0]))
    }

    /// Whether two units act on the same set of qubits.
    fn same_qubits(&self, a: u32, b: u32) -> bool {
        let (a, b) = (self.unit_wires(a), self.unit_wires(b));
        a.len() == b.len() && a.iter().all(|x| b.iter().any(|y| x.qubit == y.qubit))
    }

    /// Whether a unit is one of the units a compound was made from (a plain unit counts as its
    /// own source).
    fn is_source(&self, compound: u32, unit: u32) -> bool {
        let sources = &self.units[compound as usize].sources;
        if sources.is_empty() {
            unit == compound
        } else {
            sources.contains(&unit)
        }
    }

    fn run_len(&self, qubit: u32) -> i64 {
        let k = usize::from(qubit != self.pair[0]);
        self.runs[k].len() as i64
    }

    /// Whether `unit` commutes with every unit between it and `swap` on `qubit`, leaving out
    /// the `skip` units next to the SWAP (they are hopped across it).
    fn commutes_between(&self, qubit: u32, unit: u32, swap: u32, skip: i64) -> bool {
        let pauli = self.pauli(unit, qubit);
        let (unit_position, swap_position) =
            (self.position(qubit, unit), self.position(qubit, swap));
        let (mut low, mut high) = if unit_position < swap_position {
            (unit_position, swap_position)
        } else {
            (swap_position, unit_position)
        };
        if unit_position < swap_position {
            high -= skip;
        } else {
            low += skip;
        }
        pauli != Pauli::Unknown
            && between(&self.sequence[qubit as usize], low, high)
                .iter()
                .all(|&other| self.pauli(other, qubit) == pauli)
    }

    /// Whether the tail rule's T2 part leaves the SWAP at wire position `index` on `qubit`
    /// alone: the qubit has a short T2, is not measured after the SWAP, and its multi-qubit units
    /// after the SWAP other than `unit` all act on one set of qubits.
    fn ends_on_one_pair(&self, qubit: u32, index: u32, unit: u32) -> bool {
        if !self.short_t2[qubit as usize] || self.last_measure[qubit as usize] > index as i64 {
            return false;
        }
        let mut seen = NONE;
        for &node in &self.wires[qubit as usize][index as usize + 1..] {
            let other = self.unit_of[node as usize];
            if self.qargs(node).len() < 2
                || self.is_source(unit, other)
                || self.flags[node as usize].directive
            {
                continue;
            }
            if seen == NONE {
                seen = other;
            } else if !self.same_qubits(other, seen) {
                return false;
            }
        }
        true
    }

    /// Collect into `self.runs[k]` the single-qubit gates next to `swap` on `qubit`, towards
    /// `step`, nearest first.  `SWAP (g x I) = (I x g) SWAP` for every single-qubit gate `g`, so
    /// these can be moved to the other qubit on the other side of the SWAP, out of a unit's way.
    fn hop_run(&mut self, k: usize, swap: u32, step: i64) {
        let qubit = self.pair[k];
        let mut run = std::mem::take(&mut self.runs[k]);
        let seq = &self.sequence[qubit as usize];
        let mut index = self.position(qubit, swap) + step;
        while index >= 0 && (index as usize) < seq.len() {
            let unit = seq[index as usize];
            if self.num_qubits_of(unit) != 1 {
                break;
            }
            let node = self.units[unit as usize].members[0];
            if self.hopped[node as usize]
                || self.taken[unit as usize]
                || !self.flags[node as usize].gate
            {
                break;
            }
            run.push(node);
            index += step;
        }
        self.runs[k] = run;
    }

    /// Whether moving the unit towards `step` would split an existing same-pair two-qubit run.
    fn has_same_pair_block(&self, unit: u32, step: i64) -> bool {
        let wires = self.unit_wires(unit);
        let pair = [wires[0].qubit, wires[1].qubit];
        let mut neighbours = [NONE; 2];
        for (k, &qubit) in pair.iter().enumerate() {
            let seq = &self.sequence[qubit as usize];
            let mut index = self.position(qubit, unit) + step;
            while index >= 0
                && (index as usize) < seq.len()
                && self.num_qubits_of(seq[index as usize]) == 1
            {
                index += step;
            }
            if index >= 0 && (index as usize) < seq.len() {
                neighbours[k] = seq[index as usize];
            }
        }
        neighbours[0] != NONE && neighbours[0] == neighbours[1] && self.on_pair(neighbours[0], pair)
    }

    /// Whether another multi-qubit unit remains between the unit and the SWAP on `qubit`.
    fn gap_remains(&self, qubit: u32, unit: u32, swap: u32) -> bool {
        let (a, b) = (self.position(qubit, unit), self.position(qubit, swap));
        let (low, high) = if a < b { (a, b) } else { (b, a) };
        between(&self.sequence[qubit as usize], low, high)
            .iter()
            .any(|&other| {
                if self.num_qubits_of(other) <= 1 {
                    return false;
                }
                let destination = self.destinations[other as usize];
                if destination == NONE {
                    return true;
                }
                let position = self
                    .unit_wires(destination)
                    .iter()
                    .find(|wire| wire.qubit == qubit)
                    .map_or(-1, |wire| wire.position as i64);
                low < position && position < high
            })
    }

    /// A complete CZ block around `unit`, extended towards `step`, whose members commute with
    /// every external unit they cross on the way to `swap`, as a new compound unit; or `None`.
    fn donor_block(&mut self, unit: u32, swap: u32, step: i64) -> Option<u32> {
        let result = self.donor_block_inner(unit, swap, step);
        for &member in &self.block {
            self.in_block[member as usize] = false;
        }
        self.block.clear();
        result
    }

    fn donor_block_inner(&mut self, unit: u32, swap: u32, step: i64) -> Option<u32> {
        let all_cz = |search: &Self, unit: u32| {
            search.units[unit as usize]
                .members
                .iter()
                .all(|&member| search.flags[member as usize].cz)
        };
        if !all_cz(self, unit) {
            return None;
        }
        let wires = self.unit_wires(unit);
        let pair = [wires[0].qubit, wires[1].qubit];
        self.block.push(unit);
        self.in_block[unit as usize] = true;
        let mut cursor = unit;
        let mut proposed: SmallVec<[u32; 8]> = SmallVec::new();
        while self.has_same_pair_block(cursor, step) {
            proposed.clear();
            let mut following = NONE;
            for &qubit in &pair {
                let seq = &self.sequence[qubit as usize];
                let mut index = self.position(qubit, cursor) + step;
                while self.num_qubits_of(seq[index as usize]) == 1 {
                    proposed.push(seq[index as usize]);
                    index += step;
                }
                following = seq[index as usize];
            }
            if !all_cz(self, following) {
                return None;
            }
            proposed.push(following);
            if proposed.iter().any(|&other| {
                self.taken[other as usize]
                    || self.units[other as usize].members.iter().any(|&member| {
                        self.hopped[member as usize]
                            || !self.flags[member as usize].gate
                            || self.flags[member as usize].swap
                    })
            }) {
                return None;
            }
            for &other in &proposed {
                if !self.in_block[other as usize] {
                    self.in_block[other as usize] = true;
                    self.block.push(other);
                }
            }
            cursor = following;
        }
        for &other in &self.block {
            for wire in self.unit_wires(other) {
                let qubit = wire.qubit;
                let (a, b) = (wire.position as i64, self.position(qubit, swap));
                let (mut low, mut high) = if a < b { (a, b) } else { (b, a) };
                if a < b {
                    high -= self.run_len(qubit);
                } else {
                    low += self.run_len(qubit);
                }
                if between(&self.sequence[qubit as usize], low, high)
                    .iter()
                    .any(|&blocker| {
                        !self.in_block[blocker as usize]
                            && (wire.pauli == Pauli::Unknown
                                || self.pauli(blocker, qubit) != wire.pauli)
                    })
                {
                    return None;
                }
            }
        }
        let compound = self.units.len() as u32;
        let mut members: SmallVec<[u32; 3]> = self
            .block
            .iter()
            .flat_map(|&other| self.units[other as usize].members.iter().copied())
            .collect();
        members.sort_unstable();
        // The compound takes the base unit's Paulis and positions.
        let start = self.unit_wires.len() as u32;
        let (base_start, base_len) = self.units[unit as usize].wires;
        self.unit_wires
            .extend_from_within(base_start as usize..(base_start + base_len) as usize);
        self.units.push(Unit {
            members,
            wires: (start, base_len),
            sources: self.block.clone(),
        });
        self.taken.push(false);
        self.destinations.push(NONE);
        self.in_block.push(false);
        Some(compound)
    }

    /// The move for `swap`, or `None`; with `hop`, gates from the hop runs may be relabelled
    /// across the SWAP.
    fn search(&mut self, swap: u32, hop: bool) -> Option<Move> {
        let node = self.units[swap as usize].members[0];
        let swap_wires: [NodeWire; 2] = [self.qargs(node)[0], self.qargs(node)[1]];
        // A qubit that does no further multi-qubit work after the SWAP would end on the merged
        // block's trailing single-qubit gates, which an as-late-as-possible schedule leaves
        // waiting until the end of the circuit; leave the SWAP decomposition there.
        if swap_wires
            .iter()
            .any(|wire| self.last_multi[wire.qubit as usize] <= wire.position as i64)
        {
            return None;
        }
        self.pair = [swap_wires[0].qubit, swap_wires[1].qubit];
        let pair = self.pair;
        let qubit = pair[0];
        for step in [1, -1] {
            for k in 0..2 {
                self.runs[k].clear();
                if hop {
                    self.hop_run(k, swap, step);
                }
            }
            if hop && self.runs[0].is_empty() && self.runs[1].is_empty() {
                continue;
            }
            // Walk to the nearest unit of the pair, stopping at a unit nothing can be moved past.
            let seq = &self.sequence[qubit as usize];
            let mut index = self.position(qubit, swap) + step * (1 + self.runs[0].len() as i64);
            while index >= 0
                && (index as usize) < seq.len()
                && !self.on_pair(seq[index as usize], pair)
                && self.pauli(seq[index as usize], qubit) != Pauli::Unknown
            {
                index += step;
            }
            if index < 0 || (index as usize) >= seq.len() {
                continue;
            }
            let mut unit = seq[index as usize];
            if !self.on_pair(unit, pair)
                || self.taken[unit as usize]
                || !(0..2)
                    .all(|k| self.commutes_between(pair[k], unit, swap, self.runs[k].len() as i64))
            {
                continue;
            }
            // The same-pair-run guard applies to every move, not only to hops: a unit is not
            // pulled out of a run the peephole already merges (a CZ-only run may move whole).
            if self.has_same_pair_block(unit, step) {
                unit = self.donor_block(unit, swap, step)?;
            }
            if hop && !pair.iter().any(|&q| self.gap_remains(q, unit, swap)) {
                return None;
            }
            // A SWAP the tail rule leaves alone is not tried on its other side either.
            if swap_wires
                .iter()
                .any(|wire| self.ends_on_one_pair(wire.qubit, wire.position, unit))
            {
                return None;
            }
            // The hops of the lower-indexed qubit first; this fixes the order in which the
            // gates are removed from the DAG and has no effect on the circuit.
            let mut hops = SmallVec::new();
            let lower = usize::from(pair[0] > pair[1]);
            hops.extend_from_slice(&self.runs[lower]);
            hops.extend_from_slice(&self.runs[1 - lower]);
            return Some(Move {
                unit,
                swap,
                step,
                hops,
            });
        }
        None
    }

    fn record(&mut self, mv: Move) {
        self.taken[mv.unit as usize] = true;
        for &hop in &mv.hops {
            self.hopped[hop as usize] = true;
        }
        self.destinations[mv.unit as usize] = mv.swap;
        // A whole-block donor keeps its original units out of any later move.
        for &source in &self.units[mv.unit as usize].sources {
            self.taken[source as usize] = true;
            self.destinations[source as usize] = mv.swap;
        }
        self.moves.push(mv);
    }

    /// Find the moves: first without hops, then with hops for the SWAPs still without a move.
    fn run(&mut self) {
        let mut pending = Vec::new();
        // A SWAP already adjacent (on both wires) to another unit of its pair is left alone, and
        // so is that unit; the peephole optimization merges them as they are.
        for unit in 0..self.units.len() as u32 {
            let first = self.units[unit as usize].members[0];
            if !self.flags[first as usize].swap {
                continue;
            }
            let wires = self.unit_wires(unit);
            let pair = [wires[0].qubit, wires[1].qubit];
            let mut touching = false;
            for &qubit in &pair {
                let seq = &self.sequence[qubit as usize];
                let position = self.position(qubit, unit);
                for index in [position - 1, position + 1] {
                    if index < 0 || (index as usize) >= seq.len() {
                        continue;
                    }
                    let other = seq[index as usize];
                    if self.on_pair(other, pair)
                        && pair
                            .iter()
                            .all(|&q| (self.position(q, other) - self.position(q, unit)).abs() == 1)
                    {
                        self.taken[other as usize] = true;
                        touching = true;
                    }
                }
            }
            if !touching {
                pending.push(unit);
            }
        }
        for &swap in &pending {
            if let Some(mv) = self.search(swap, false) {
                self.record(mv);
            }
        }
        let mut moved = vec![false; self.units.len()];
        for mv in &self.moves {
            moved[mv.swap as usize] = true;
        }
        for &swap in &pending {
            if !moved[swap as usize]
                && let Some(mv) = self.search(swap, true)
            {
                self.record(mv);
            }
        }
    }
}

/// Find the moves for `dag` without applying them.
pub fn find_moves(dag: &DAGCircuit, short_t2: &[u32]) -> Plan {
    let order: Vec<NodeIndex> = dag.topological_op_nodes(false).collect();
    let mut search = MoveSearch::new(dag, &order, short_t2);
    search.run();
    Plan {
        order,
        units: search.units,
        unit_wires: search.unit_wires,
        moves: search.moves,
    }
}

/// Apply the moves of `plan` to `dag`: every moved unit is re-applied right next to its SWAP,
/// with the hopped gates on the SWAP's other side.
pub fn apply_moves(dag: &mut DAGCircuit, plan: &Plan) -> Result<(), DAGError> {
    let mut removed: HashMap<u32, PackedInstruction> = HashMap::new();
    for mv in &plan.moves {
        for &node in plan.units[mv.unit as usize]
            .members
            .iter()
            .chain(mv.hops.iter())
        {
            removed.insert(node, dag.remove_op_node(plan.order[node as usize]));
        }
    }
    for mv in &plan.moves {
        let swap_index = plan.order[plan.units[mv.swap as usize].members[0] as usize];
        let swap = dag[swap_index].unwrap_operation();
        let swap_qargs: [Qubit; 2] = {
            let qargs = dag.get_qargs(swap.qubits);
            [qargs[0], qargs[1]]
        };
        let members = &plan.units[mv.unit as usize].members;
        let mut hops = mv.hops.clone();
        hops.sort_unstable();
        let mut block =
            DAGCircuit::with_capacity(2, 0, None, Some(members.len() + hops.len() + 1), None, None);
        for _ in 0..2 {
            block.add_qubit_unchecked(ShareableQubit::new_anonymous())?;
        }
        // Hopped gates go onto the SWAP's other qubit; everything else keeps its qubits.
        let add = |block: &mut DAGCircuit,
                   inst: PackedInstruction,
                   qargs: &[Qubit],
                   reversed: bool|
         -> Result<(), DAGError> {
            let mapped: SmallVec<[Qubit; 2]> = qargs
                .iter()
                .map(|qubit| {
                    let index = usize::from(*qubit != swap_qargs[0]);
                    Qubit::new(if reversed { 1 - index } else { index })
                })
                .collect();
            block.apply_operation_back(
                inst.op,
                &mapped,
                &[],
                inst.params.map(|params| *params),
                inst.label.map(|label| *label),
                #[cfg(feature = "cache_pygates")]
                inst.py_op.into_inner(),
            )?;
            Ok(())
        };
        let mut add_removed = |block: &mut DAGCircuit, node: u32, reversed: bool| {
            let inst = removed.remove(&node).expect("moved operation was removed");
            let qargs: SmallVec<[Qubit; 2]> = dag.get_qargs(inst.qubits).iter().copied().collect();
            add(block, inst, &qargs, reversed)
        };
        let swap_inst = swap.clone();
        if mv.step > 0 {
            for &hop in &hops {
                add_removed(&mut block, hop, true)?;
            }
            add(&mut block, swap_inst, &swap_qargs, false)?;
            for &member in members {
                add_removed(&mut block, member, false)?;
            }
        } else {
            for &member in members {
                add_removed(&mut block, member, false)?;
            }
            add(&mut block, swap_inst, &swap_qargs, false)?;
            for &hop in &hops {
                add_removed(&mut block, hop, true)?;
            }
        }
        let qubit_map: HashMap<Qubit, Qubit> = HashMap::from_iter([
            (Qubit::new(0), swap_qargs[0]),
            (Qubit::new(1), swap_qargs[1]),
        ]);
        dag.substitute_node_with_dag(
            swap_index,
            &block,
            Some(&qubit_map),
            Some(&HashMap::new()),
            Some(&HashMap::new()),
            Some(&HashMap::new()),
        )?;
    }
    Ok(())
}

/// Run the SWAP absorption on `dag` in place.  `short_t2` lists the qubits the T2 part of the
/// tail rule applies to.  Returns the number of moves made.
pub fn run_absorb_into_swaps(dag: &mut DAGCircuit, short_t2: &[u32]) -> Result<usize, DAGError> {
    if !dag.get_op_counts().contains_key("swap") {
        return Ok(0);
    }
    let plan = find_moves(dag, short_t2);
    if !plan.moves.is_empty() {
        apply_moves(dag, &plan)?;
    }
    Ok(plan.moves.len())
}

/// Run the SWAP absorption on `dag` in place.
///
/// Args:
///     dag (DAGCircuit): the DAG to rewrite.
///     short_t2 (list[int]): the qubits the T2 part of the tail rule applies to.
///
/// Returns:
///     int: the number of moves made.
#[pyfunction]
#[pyo3(name = "absorb_into_swaps", signature = (dag, short_t2))]
pub fn py_run_absorb_into_swaps(dag: &mut PyDAGCircuit, short_t2: Vec<u32>) -> PyResult<usize> {
    run_absorb_into_swaps(dag.try_write()?, &short_t2).map_err(Into::into)
}

/// The units and moves the pass would make on `dag`, without applying them.
///
/// Operations are referred to by their position in ``dag.topological_op_nodes()``.  Each unit
/// is ``(members, qubits, paulis)``: the positions of its operations, its qubits, and the
/// Pauli it commutes with on each of them (``"-"`` for none).  Each move is
/// ``(unit, swap, step, hops)``: the unit, the SWAP unit it moves next to, ``1`` for the side
/// after the SWAP and ``-1`` for the side before it, and the positions of the single-qubit
/// gates relabelled across the SWAP.
#[pyfunction]
#[pyo3(name = "swap_absorption_moves", signature = (dag, short_t2))]
#[allow(clippy::type_complexity)]
pub fn py_swap_absorption_moves(
    dag: &PyDAGCircuit,
    short_t2: Vec<u32>,
) -> PyResult<(
    Vec<(Vec<u32>, Vec<u32>, String)>,
    Vec<(u32, u32, i64, Vec<u32>)>,
)> {
    let plan = find_moves(dag.try_read()?, &short_t2);
    let units = plan
        .units
        .iter()
        .map(|unit| {
            let (start, len) = unit.wires;
            let wires = &plan.unit_wires[start as usize..(start + len) as usize];
            (
                unit.members.to_vec(),
                wires.iter().map(|wire| wire.qubit).collect(),
                wires.iter().map(|wire| wire.pauli.as_char()).collect(),
            )
        })
        .collect();
    let moves = plan
        .moves
        .iter()
        .map(|mv| (mv.unit, mv.swap, mv.step, mv.hops.to_vec()))
        .collect();
    Ok((units, moves))
}

pub fn swap_absorption_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_absorb_into_swaps))?;
    m.add_wrapped(wrap_pyfunction!(py_swap_absorption_moves))?;
    Ok(())
}
