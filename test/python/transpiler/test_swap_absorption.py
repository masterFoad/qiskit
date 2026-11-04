# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the pass that moves two-qubit gates next to routing SWAPs."""

import itertools
import random
from unittest.mock import patch

import numpy as np
from ddt import ddt, data, idata, unpack

from qiskit.circuit import Gate, Parameter, QuantumCircuit
from qiskit.circuit.library import QFTGate, UnitaryGate, qaoa_ansatz
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap, PassManagerConfig, Target, generate_preset_pass_manager
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager
from qiskit.transpiler.preset_passmanagers import _swap_absorption, common
from qiskit.transpiler.preset_passmanagers._swap_absorption import _AbsorbIntoSwaps
from test import QiskitTestCase


@ddt
class TestAbsorbIntoSwaps(QiskitTestCase):
    """Test the ``_AbsorbIntoSwaps`` pass on small circuits.

    The circuits act on at most 4 qubits, so each ``Operator`` is at most 16 x 16.
    """

    @data("cp", "cz", "rzz", "cx", "rxx", "ryy", "diagonal unitary", "cx-rz-cx")
    def test_gate_moves_next_to_swap(self, kind):
        """A gate on the SWAP's qubits moves through operations it commutes with."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        if kind == "cp":
            qc.rz(0.2, 0)
            qc.cz(0, 2)
            qc.cp(0.3, 0, 1)
        elif kind == "cz":
            qc.t(1)
            qc.cp(0.4, 1, 2)
            qc.cz(0, 1)
        elif kind == "rzz":
            qc.rz(0.2, 1)
            qc.rzz(0.4, 1, 2)
            qc.rzz(0.3, 0, 1)
        elif kind == "cx":
            qc.rz(0.2, 0)  # diagonal on the control
            qc.sx(1)  # X-like on the target
            qc.cx(0, 1)
        elif kind == "rxx":
            qc.rx(0.3, 0)
            qc.rxx(0.5, 0, 1)
        elif kind == "ryy":
            qc.ry(0.3, 0)
            qc.ryy(0.2, 1, 2)
            qc.ryy(0.5, 0, 1)
        elif kind == "diagonal unitary":
            qc.p(0.1, 1)
            qc.cp(0.5, 1, 2)
            qc.append(UnitaryGate(np.diag([1, 1j, -1, -1j])), [0, 1])
        else:
            qc.cz(1, 2)
            qc.cx(0, 1)
            qc.rz(0.7, 1)
            qc.cx(0, 1)
        # More two-qubit gates on both SWAP qubits, so that the SWAP is not at the end of a wire.
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        moved = {"diagonal unitary": "unitary", "cx-rz-cx": "cx"}.get(kind, kind)
        dag = circuit_to_dag(out)
        (swap,) = dag.named_nodes("swap")
        self.assertEqual({node.name for node in dag.op_successors(swap)}, {moved})
        self.assertEqual(dict(out.count_ops()), dict(qc.count_ops()))
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_earlier_gate_moves_forward(self):
        """A gate before the SWAP moves forward to it."""
        qc = QuantumCircuit(3)
        qc.cp(0.3, 0, 1)
        qc.cz(0, 2)
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        dag = circuit_to_dag(out)
        (swap,) = dag.named_nodes("swap")
        self.assertEqual({node.name for node in dag.op_predecessors(swap)}, {"cp"})
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(
        "h",
        "barrier",
        "small ry",
        "rz on cx target",
        "control flow",
        "custom gate named rz",
        "custom gate named swap",
        "nearly diagonal unitary",
        "swap at end of wire",
        "swap already next to a gate",
    )
    def test_no_move(self, kind):
        """The circuit is unchanged when the move is not allowed."""
        qc = QuantumCircuit(3, 1)
        interaction = "cp"
        if kind == "swap at end of wire":
            # No multi-qubit operation follows the SWAP on qubit 1.
            qc.cp(0.3, 0, 1)
            qc.cz(0, 2)
            qc.swap(0, 1)
            qc.cx(0, 2)
            qc.h(1)
        else:
            if kind == "custom gate named swap":
                definition = QuantumCircuit(2)
                definition.cx(0, 1)
                fake_swap = Gate("swap", 2, [])
                fake_swap.definition = definition
                qc.append(fake_swap, [0, 1])
            else:
                if kind == "swap already next to a gate":
                    qc.cz(0, 1)
                qc.swap(0, 1)
            if kind == "h":
                qc.h(0)
            elif kind == "barrier":
                qc.barrier(0, 1)
            elif kind == "small ry":
                qc.ry(1e-6, 0)
            elif kind == "rz on cx target":
                qc.rz(0.2, 1)
                interaction = "cx"
            elif kind == "control flow":
                with qc.if_test((qc.clbits[0], 1)):
                    qc.rz(0.2, 0)
            elif kind == "custom gate named rz":
                definition = QuantumCircuit(1)
                definition.h(0)
                fake_rz = Gate("rz", 1, [])
                fake_rz.definition = definition
                qc.append(fake_rz, [0])
            elif kind == "nearly diagonal unitary":
                matrix = np.diag([1, 1j, -1, -1j]).astype(complex)
                matrix[0, 1] = matrix[1, 0] = 1e-13
                qc.append(UnitaryGate(matrix, check_input=False), [0, 2])
            else:
                qc.rz(0.2, 0)
            if interaction == "cx":
                qc.cx(0, 1)
            else:
                qc.cp(0.3, 0, 1)
            qc.cx(0, 2)
            qc.cx(1, 2)
        qc.ensure_physical()

        self.assertEqual(_AbsorbIntoSwaps()(qc), qc)

    def test_missing_table_entry_blocks_move(self):
        """Without a stored commutation relation, nothing moves."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ensure_physical()
        with patch.object(
            _swap_absorption.SessionCommutationChecker,
            "check_commutation_entries",
            return_value=None,
        ):
            self.assertEqual(_AbsorbIntoSwaps()(qc), qc)

    def test_not_moved_across_another_swap(self):
        """A gate is not moved to a SWAP across a SWAP on another pair that shares a qubit."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.swap(1, 2)
        qc.cp(0.3, 0, 1)
        qc.cx(0, 2)
        qc.cz(1, 2)
        qc.ensure_physical()

        self.assertEqual(_AbsorbIntoSwaps()(qc), qc)

    def test_gate_between_two_swaps_used_once(self):
        """A gate that both of two SWAPs on its pair could take moves to only one of them."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.cz(0, 2)
        qc.cp(0.3, 0, 1)
        qc.cz(1, 2)
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(dict(out.count_ops()), dict(qc.count_ops()))
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))
        dag = circuit_to_dag(out)
        (cp,) = dag.named_nodes("cp")
        before = {node.name for node in dag.op_predecessors(cp)}
        after = {node.name for node in dag.op_successors(cp)}
        self.assertIn({"swap"}, (before, after))

    def test_gate_on_each_side_of_swap(self):
        """With a movable gate on each side of a SWAP, one of them moves next to it."""
        qc = QuantumCircuit(3)
        qc.cp(0.3, 0, 1)
        qc.cz(0, 2)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cz(1, 2)
        qc.rzz(0.4, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        dag = circuit_to_dag(out)
        (swap,) = dag.named_nodes("swap")
        before = {node.name for node in dag.op_predecessors(swap)}
        after = {node.name for node in dag.op_successors(swap)}
        self.assertTrue(before == {"cp"} or after == {"rzz"})
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_not_moved_if_it_splits_a_run(self):
        """A gate is not moved away from a later gate on the same pair that it is next to."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cp(0.3, 0, 1)
        qc.cz(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(out, qc)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_cx_diagonal_cx_needs_adjacent_control(self):
        """``cx; D; cx`` is not moved as one gate if another gate acts on the control between."""
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.p(0.1, 1)
        qc.cx(0, 1)
        qc.rz(0.3, 1)
        qc.rz(0.4, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(out, qc)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_barrier_after_swap_allows_move(self):
        """A barrier after the SWAP fences the qubits, so the gate may move to the SWAP."""
        qc = QuantumCircuit(3)
        qc.cp(0.3, 0, 1)
        qc.cz(0, 2)
        qc.swap(0, 1)
        qc.cx(0, 2)
        qc.h(1)
        qc.barrier()
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        dag = circuit_to_dag(out)
        (swap,) = dag.named_nodes("swap")
        self.assertEqual({node.name for node in dag.op_predecessors(swap)}, {"cp"})
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_moves_at_two_swaps(self):
        """At two SWAPs that share a qubit, the ``cp`` moves back to the first SWAP and the
        ``rzz`` moves forward to the second, each across the other."""
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.swap(0, 1)
        qc.rz(0.13, 1)
        qc.rzz(0.41, 1, 2)
        qc.rz(0.29, 1)
        qc.cp(-0.37, 0, 1)
        qc.swap(1, 2)
        qc.cx(2, 3)
        qc.cx(0, 1)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))
        expected = QuantumCircuit(4)
        expected.h(range(4))
        expected.swap(0, 1)
        expected.cp(-0.37, 0, 1)
        expected.rz(0.13, 1)
        expected.rz(0.29, 1)
        expected.rzz(0.41, 1, 2)
        expected.swap(1, 2)
        expected.cx(2, 3)
        expected.cx(0, 1)
        expected.ensure_physical()
        self.assertEqual(out, expected)

    def test_parameterized_gate(self):
        """A parameterized ``cp`` moves and keeps its parameter."""
        theta = Parameter("theta")
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.rz(0.2, 0)
        qc.cp(theta, 0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        dag = circuit_to_dag(out)
        (swap,) = dag.named_nodes("swap")
        self.assertEqual({node.name for node in dag.op_successors(swap)}, {"cp"})
        self.assertEqual(out.parameters, qc.parameters)
        out = out.assign_parameters([0.37])
        qc = qc.assign_parameters([0.37])
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_random_circuits(self):
        """Random 3- and 4-qubit circuits with SWAPs keep their operator."""
        rng = random.Random(2026)
        changed = 0
        for _ in range(200):
            num_qubits = rng.choice((3, 4))
            qc = QuantumCircuit(num_qubits)
            for _ in range(rng.randint(8, 18)):
                if rng.random() < 0.3:
                    name = rng.choice(("rz", "sx", "h", "s", "x"))
                    qubit = rng.randrange(num_qubits)
                    getattr(qc, name)(*((rng.uniform(-1, 1),) if name == "rz" else ()), qubit)
                    continue
                name = rng.choice(("cx", "cz", "cp", "rzz", "rxx", "swap", "swap", "cx-rz-cx"))
                a, b = rng.sample(range(num_qubits), 2)
                if name == "cx-rz-cx":
                    qc.cx(a, b)
                    qc.rz(rng.uniform(-1, 1), b)
                    qc.cx(a, b)
                elif name in ("cp", "rzz", "rxx"):
                    getattr(qc, name)(rng.uniform(-1, 1), a, b)
                else:
                    getattr(qc, name)(a, b)
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            qc.ensure_physical()

            out = _AbsorbIntoSwaps()(qc)

            changed += out != qc
            self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))
        # The sample must exercise the pass, not only the unchanged path.
        self.assertGreater(changed, 0)


@ddt
class TestAbsorbIntoSwapsPresets(QiskitTestCase):
    """Test the pass in the preset pass managers.

    The ``Operator`` checks are on circuits of at most 6 qubits, so each ``Operator`` is at most
    64 x 64.
    """

    @idata(itertools.product([0, 1, 2, 3], ["translator", "synthesis"]))
    @unpack
    def test_runs_at_levels_2_and_3(self, level, translation_method):
        """The pass runs in the translation stage at levels 2 and 3 only."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, range(1, 4))
        target = Target.from_configuration(
            ["cz", "rz", "sx", "x"], num_qubits=4, coupling_map=CouplingMap.from_line(4)
        )
        passes = []
        pm = generate_preset_pass_manager(
            level, target=target, translation_method=translation_method
        )
        pm.run(qc, callback=lambda **kwargs: passes.append(type(kwargs["pass_"]).__name__))
        self.assertEqual("_AbsorbIntoSwaps" in passes, level in (2, 3))

    @idata(itertools.product([None, 0, 1, 2, 3], ["default", "translator", "synthesis"]))
    @unpack
    def test_translation_plugins(self, level, method):
        """The built-in translation plugins add the pass at levels 2 and 3 only."""
        target = Target.from_configuration(
            ["cz", "rz", "sx", "x"], num_qubits=2, coupling_map=CouplingMap.from_line(2)
        )
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        passes = []
        pm = PassManagerStagePluginManager().get_passmanager_stage(
            "translation", method, PassManagerConfig(target=target), optimization_level=level
        )
        pm.run(qc, callback=lambda **kwargs: passes.append(type(kwargs["pass_"]).__name__))
        self.assertEqual("_AbsorbIntoSwaps" in passes, level in (2, 3))

    @data(None, 0, 1, 2, 3)
    def test_translation_passmanager_level(self, level):
        """``generate_translation_passmanager`` adds the pass only for level 2 or 3."""
        target = Target.from_configuration(
            ["cz", "rz", "sx", "x"], num_qubits=2, coupling_map=CouplingMap.from_line(2)
        )
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        passes = []
        pm = common.generate_translation_passmanager(target, optimization_level=level)
        pm.run(qc, callback=lambda **kwargs: passes.append(type(kwargs["pass_"]).__name__))
        self.assertEqual("_AbsorbIntoSwaps" in passes, level in (2, 3))

    @data(2, 3)
    def test_qaoa_on_line(self, level):
        """Routed QAOA on a line needs fewer two-qubit gates than without the pass."""
        cost = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, j], 1.0) for i in range(5) for j in range(i + 1, 5)], num_qubits=5
        )
        qc = qaoa_ansatz(cost, reps=1).assign_parameters([0.4, 0.9])
        target = Target.from_configuration(
            ["cz", "rz", "sx", "x"], num_qubits=5, coupling_map=CouplingMap.from_line(5)
        )

        out = generate_preset_pass_manager(level, target=target, seed_transpiler=7).run(qc)
        with patch.object(_AbsorbIntoSwaps, "run", lambda self, dag: dag):
            control = generate_preset_pass_manager(level, target=target, seed_transpiler=7).run(qc)

        self.assertLess(out.count_ops()["cz"], control.count_ops()["cz"])
        check_map = CheckMap(target)
        check_map(out)
        self.assertTrue(check_map.property_set["is_swap_mapped"])
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(2, 3)
    def test_dense_crz_on_line(self, level):
        """Controlled rotations in both directions between all pairs of 4 qubits compile exactly."""
        qc = QuantumCircuit(4)
        qc.h(range(4))
        for i in range(4):
            for j in range(4):
                if i != j:
                    qc.crz(0.1 * (i + 2 * j + 1), i, j)
        target = Target.from_configuration(
            ["cz", "rz", "sx", "x"], num_qubits=4, coupling_map=CouplingMap.from_line(4)
        )

        out = generate_preset_pass_manager(level, target=target, seed_transpiler=3).run(qc)

        check_map = CheckMap(target)
        check_map(out)
        self.assertTrue(check_map.property_set["is_swap_mapped"])
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @idata(itertools.product([2, 3], [True, False]))
    @unpack
    def test_basis_gates_without_target(self, level, with_coupling_map):
        """With ``basis_gates`` instead of a target the pass runs and the result is exact."""
        qc = QuantumCircuit(4)
        qc.append(QFTGate(4), range(4))
        passes = []
        pm = generate_preset_pass_manager(
            level,
            basis_gates=["cz", "rz", "sx", "x"],
            coupling_map=CouplingMap.from_line(4) if with_coupling_map else None,
            seed_transpiler=1,
        )

        out = pm.run(qc, callback=lambda **kwargs: passes.append(type(kwargs["pass_"]).__name__))

        self.assertIn("_AbsorbIntoSwaps", passes)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_level_3_initial_layout_kept(self):
        """At level 3 an ``initial_layout`` is kept: the layout equals that without the pass."""
        cost = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, j], 1.0) for i in range(5) for j in range(i + 1, 5)], num_qubits=5
        )
        qc = qaoa_ansatz(cost, reps=1).assign_parameters([0.4, 0.9])
        target = Target.from_configuration(
            ["cz", "rz", "sx", "x"], num_qubits=5, coupling_map=CouplingMap.from_line(5)
        )
        options = {"target": target, "initial_layout": [4, 3, 2, 1, 0], "seed_transpiler": 7}

        out = generate_preset_pass_manager(3, **options).run(qc)
        with patch.object(_AbsorbIntoSwaps, "run", lambda self, dag: dag):
            control = generate_preset_pass_manager(3, **options).run(qc)

        self.assertNotEqual(out, control)
        self.assertEqual(out.layout, control.layout)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(2, 3)
    def test_qft_on_line(self, level):
        """A QFT on a line keeps its operator, global phase included."""
        qc = QuantumCircuit(6)
        qc.append(QFTGate(6), range(6))
        target = Target.from_configuration(
            ["cz", "rz", "sx", "x"], num_qubits=6, coupling_map=CouplingMap.from_line(6)
        )

        out = generate_preset_pass_manager(level, target=target, seed_transpiler=1).run(qc)

        check_map = CheckMap(target)
        check_map(out)
        self.assertTrue(check_map.property_set["is_swap_mapped"])
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_level_2_layout_unchanged(self):
        """At level 2 the pass runs after the layout is fixed, so the layout is unchanged."""
        backend = GenericBackendV2(
            num_qubits=19,
            basis_gates=["cz", "rz", "sx", "x"],
            coupling_map=CouplingMap.from_heavy_hex(3),
            seed=7,
        )
        qc = QuantumCircuit(8)
        qc.append(QFTGate(8), range(8))
        qc.measure_all()

        out = generate_preset_pass_manager(2, backend=backend, seed_transpiler=1).run(qc)
        with patch.object(_AbsorbIntoSwaps, "run", lambda self, dag: dag):
            control = generate_preset_pass_manager(2, backend=backend, seed_transpiler=1).run(qc)

        self.assertNotEqual(out, control)
        self.assertEqual(out.layout, control.layout)
