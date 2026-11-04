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
from qiskit.circuit.library import QFTGate, qaoa_ansatz
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap, PassManagerConfig, Target, generate_preset_pass_manager
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager
from qiskit.transpiler.preset_passmanagers import _swap_absorption, common
from qiskit.transpiler.preset_passmanagers._swap_absorption import _AbsorbIntoSwaps
from test import QiskitTestCase


def _circuit(num_qubits, gates, num_clbits=0):
    """Build a circuit from ``(method, *args)`` rows."""
    qc = QuantumCircuit(num_qubits, num_clbits)
    for method, *args in gates:
        getattr(qc, method)(*args)
    return qc


def _custom_gate(name, num_qubits, gates):
    """A custom gate with the given name and a definition built from ``gates``."""
    gate = Gate(name, num_qubits, [])
    gate.definition = _circuit(num_qubits, gates)
    return gate


# Two-qubit gates after the SWAP on both of its qubits, so that it is not at the end of a wire.
_TAIL = [("cx", 0, 2), ("cx", 1, 2)]
_FAKE_RZ = _custom_gate("rz", 1, [("h", 0)])
_FAKE_SWAP = _custom_gate("swap", 2, [("cx", 0, 1)])
_RZ_BODY = _circuit(1, [("rz", 0.2, 0)], num_clbits=1)


@ddt
class TestAbsorbIntoSwaps(QiskitTestCase):
    """Test the ``_AbsorbIntoSwaps`` pass on small circuits.

    The circuits act on at most 4 qubits, so each ``Operator`` is at most 16 x 16.
    """

    @data(
        ("rxx", [("swap", 0, 1), ("rx", 0.3, 0), ("rxx", 0.5, 0, 1), *_TAIL]),
        ("ryy", [("swap", 0, 1), ("ry", 0.3, 0), ("ryy", 0.2, 1, 2), ("ryy", 0.5, 0, 1), *_TAIL]),
    )
    @unpack
    def test_gate_moves_next_to_swap(self, moved, gates):
        """A gate on the SWAP's qubits moves through operations it commutes with."""
        qc = _circuit(3, gates)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        dag = circuit_to_dag(out)
        (swap,) = dag.named_nodes("swap")
        self.assertEqual({node.name for node in dag.op_successors(swap)}, {moved})
        self.assertEqual(dict(out.count_ops()), dict(qc.count_ops()))
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(
        [("rxx", 0.3, 0, 1), ("rx", 0.2, 0), ("swap", 0, 1), *_TAIL],
        # A barrier after the SWAP counts as a multi-qubit operation on both of its qubits.
        [("rxx", 0.3, 0, 1), ("rx", 0.2, 0), ("swap", 0, 1), ("cx", 0, 2), ("h", 1), ("barrier",)],
    )
    def test_earlier_gate_moves_forward(self, gates):
        """A gate before the SWAP moves forward to it."""
        qc = _circuit(3, gates)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        dag = circuit_to_dag(out)
        (swap,) = dag.named_nodes("swap")
        self.assertEqual({node.name for node in dag.op_predecessors(swap)}, {"rxx"})
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(
        ([("swap", 0, 1), ("rz", 0.2, 0), ("rzz", 0.4, 0, 1), *_TAIL], {"cx": 5, "p": 1, "rz": 1}),
        ([("rzz", 0.4, 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL], {"cx": 5, "p": 1, "rz": 1}),
        ([("swap", 0, 1), ("rz", 0.2, 0), ("cp", 0.3, 0, 1), *_TAIL], {"cx": 5, "p": 3, "rz": 1}),
        ([("cp", 0.3, 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL], {"cx": 5, "p": 3, "rz": 1}),
        ([("swap", 0, 1), ("rz", 0.2, 0), ("cz", 1, 0), *_TAIL], {"cx": 5, "p": 3, "rz": 1}),
        ([("cz", 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL], {"cx": 5, "p": 3, "rz": 1}),
        (
            [("swap", 0, 1), ("rz", 0.2, 0), ("unitary", np.diag([1, 1j, -1, -1j]), [0, 1])]
            + [("barrier",)],
            {"cx": 3, "p": 2, "rz": 1, "barrier": 1},
        ),
        (
            [("unitary", np.diag([1, 1j, -1, -1j]), [1, 0]), ("rz", 0.2, 0), ("swap", 0, 1)]
            + _TAIL,
            {"cx": 5, "p": 2, "rz": 1},
        ),
        (
            [("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1), ("rz", 0.7, 1), ("cx", 0, 1), *_TAIL],
            {"cx": 5, "p": 1, "rz": 1},
        ),
        (
            [("cx", 0, 1), ("rz", 0.7, 1), ("cx", 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL],
            {"cx": 5, "p": 1, "rz": 1},
        ),
        ([("swap", 0, 1), ("rz", 0.2, 0), ("crz", 0.4, 1, 0), *_TAIL], {"cx": 5, "p": 2, "rz": 1}),
        ([("crz", 0.4, 1, 0), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL], {"cx": 5, "p": 2, "rz": 1}),
        ([("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1), *_TAIL], {"cx": 4, "rz": 1}),
        ([("cx", 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL], {"cx": 4, "rz": 1}),
        ([("swap", 0, 1), ("sx", 0), ("cx", 1, 0), *_TAIL], {"cx": 4, "sx": 1}),
        ([("cx", 1, 0), ("sx", 0), ("swap", 0, 1), *_TAIL], {"cx": 4, "sx": 1}),
    )
    @unpack
    def test_fused_forms(self, gates, counts):
        """A SWAP and a diagonal gate or a CX are written as an exact short circuit."""
        qc = _circuit(3, gates)
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(dict(out.count_ops()), counts)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(
        ([("swap", 0, 1), ("rz", 0.2, 0), ("rzz", 0.4, 0, 1), *_TAIL], True),
        (
            [("cx", 0, 1), ("rz", 0.7, 1), ("cx", 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL],
            True,
        ),
        ([("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1), *_TAIL], False),
    )
    @unpack
    def test_move_diagonal_false(self, gates, diagonal):
        """With ``move_diagonal=False`` a diagonal unit stays where it is and a CX still moves."""
        qc = _circuit(3, gates)
        qc.ensure_physical()

        moved = _AbsorbIntoSwaps()(qc)
        out = _AbsorbIntoSwaps(move_diagonal=False)(qc)

        self.assertNotEqual(moved, qc)
        self.assertEqual(out, qc if diagonal else moved)

    @data(
        ("h", [("swap", 0, 1), ("h", 0), ("cp", 0.3, 0, 1), *_TAIL]),
        ("barrier", [("swap", 0, 1), ("rz", 0.2, 0), ("barrier", 0), ("cp", 0.3, 0, 1), *_TAIL]),
        ("barrier", [("swap", 0, 1), ("rz", 0.2, 0), ("barrier", 0, 1), ("cp", 0.3, 0, 1), *_TAIL]),
        # An angle small enough that an approximate commutation check would accept it; the pass
        # must not.
        ("small ry", [("swap", 0, 1), ("ry", 1e-6, 0), ("cp", 0.3, 0, 1), *_TAIL]),
        ("rz on cx target", [("swap", 0, 1), ("rz", 0.2, 1), ("cx", 0, 1), *_TAIL]),
        (
            "cx; sx; cx",
            [("swap", 0, 1), ("rz", 0.2, 1), ("cx", 0, 1), ("sx", 1), ("cx", 0, 1), *_TAIL],
        ),
        (
            "cx; D; cx with a gate on the control",
            [("swap", 0, 1), ("rz", 0.2, 0), ("p", 0.1, 1), ("cx", 0, 1), ("rz", 0.3, 1)]
            + [("rz", 0.4, 0), ("cx", 0, 1), *_TAIL],
        ),
        (
            "control flow",
            [("swap", 0, 1), ("if_test", (0, 1), _RZ_BODY, [0], [0]), ("cp", 0.3, 0, 1), *_TAIL],
        ),
        ("custom rz", [("swap", 0, 1), ("append", _FAKE_RZ, [0]), ("cp", 0.3, 0, 1), *_TAIL]),
        (
            "custom swap",
            [("append", _FAKE_SWAP, [0, 1]), ("rz", 0.2, 0), ("cp", 0.3, 0, 1), *_TAIL],
        ),
        # Nearly diagonal; the pass only moves exactly diagonal unitaries.
        (
            "nearly diagonal unitary",
            [("swap", 0, 1)]
            + [("unitary", np.diag([1, 1j, -1, -1j]) + np.diag([1e-13, 0, 0], 1), [0, 2])]
            + [("cp", 0.3, 0, 1), *_TAIL],
        ),
        (
            "swap at end of wire",
            [("cp", 0.3, 0, 1), ("cz", 0, 2), ("swap", 0, 1), ("cx", 0, 2), ("h", 1)],
        ),
        (
            "already next to a gate",
            [("cz", 0, 1), ("swap", 0, 1), ("rz", 0.2, 0), ("cp", 0.3, 0, 1), *_TAIL],
        ),
        (
            "across another swap",
            [("swap", 0, 1), ("swap", 1, 2), ("cp", 0.3, 0, 1), ("cx", 0, 2), ("cz", 1, 2)],
        ),
        ("splits a run", [("swap", 0, 1), ("rz", 0.2, 0), ("cp", 0.3, 0, 1), ("cz", 0, 1), *_TAIL]),
    )
    @unpack
    def test_no_move(self, name, gates):
        """The circuit is unchanged when the move is not allowed."""
        qc = _circuit(3, gates, num_clbits=1)
        qc.ensure_physical()

        self.assertEqual(_AbsorbIntoSwaps()(qc), qc, name)

    def test_missing_table_entry_blocks_move(self):
        """Without a stored commutation relation, nothing moves."""
        qc = _circuit(3, [("swap", 0, 1), ("rz", 0.2, 0), ("cp", 0.3, 0, 1), *_TAIL])
        qc.ensure_physical()
        with patch.object(
            _swap_absorption.SessionCommutationChecker,
            "check_commutation_entries",
            return_value=None,
        ):
            self.assertEqual(_AbsorbIntoSwaps()(qc), qc)

    def test_gate_between_two_swaps_used_once(self):
        """A gate that both of two SWAPs on its pair could take moves to only one of them."""
        qc = _circuit(
            3,
            [("swap", 0, 1), ("cz", 0, 2), ("cp", 0.3, 0, 1), ("cz", 1, 2), ("swap", 0, 1), *_TAIL],
        )
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(dict(out.count_ops()), {"cx": 5, "p": 3, "cz": 2, "swap": 1})
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_gate_on_each_side_of_swap(self):
        """With a movable gate on each side of a SWAP, one of them is merged with it."""
        qc = _circuit(
            3,
            [("cp", 0.3, 0, 1), ("cz", 0, 2), ("swap", 0, 1), ("rz", 0.2, 0), ("cz", 1, 2)]
            + [("rzz", 0.4, 0, 1), *_TAIL],
        )
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        rzz_merged = {"cp": 1, "cz": 2, "rz": 1, "cx": 5, "p": 1}
        cp_merged = {"rzz": 1, "cz": 2, "rz": 1, "cx": 5, "p": 3}
        self.assertIn(dict(out.count_ops()), (rzz_merged, cp_merged))
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

        self.assertEqual(dict(out.count_ops()), {"h": 4, "cx": 8, "p": 4, "rz": 2})
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_parameterized_gate(self):
        """A parameterized ``cp`` moves and keeps its parameter."""
        theta = Parameter("theta")
        qc = _circuit(3, [("swap", 0, 1), ("rz", 0.2, 0), ("cp", theta, 0, 1), *_TAIL])
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
        for index in range(200):
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
            with self.subTest(index=index):
                self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))
        # The sample must exercise the pass, not only the unchanged path.
        self.assertGreater(changed, 0)

    @data(
        # Other angle.
        [("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1), ("rz", 0.7, 1), ("cx", 0, 1)]
        + [("swap", 2, 3), ("rz", 0.2, 2), ("cx", 2, 3), ("rz", 0.3, 3), ("cx", 2, 3)],
        # Other gate.
        [("swap", 0, 1), ("rz", 0.2, 0), ("crz", 0.7, 0, 1)]
        + [("swap", 2, 3), ("rz", 0.2, 2), ("cp", 0.7, 2, 3)],
        # Other qubit order.
        [("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1)]
        + [("swap", 2, 3), ("rz", 0.2, 3), ("cx", 3, 2)],
        # Other direction.
        [("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1)]
        + [("cx", 2, 3), ("rz", 0.2, 2), ("swap", 2, 3)],
    )
    def test_fused_forms_of_similar_units(self, gates):
        """Units that differ only in an angle, a gate, a qubit order or a direction are exact."""
        qc = _circuit(4, gates + [("cx", 0, 2), ("cx", 1, 3)])
        qc.ensure_physical()

        out = _AbsorbIntoSwaps()(qc)

        self.assertNotIn("swap", out.count_ops())
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))


@ddt
class TestAbsorbIntoSwapsPresets(QiskitTestCase):
    """Test the pass in the preset pass managers.

    The ``Operator`` checks are on circuits of at most 6 qubits, so each ``Operator`` is at most
    64 x 64.
    """

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

    @data(
        (None, ["cz", "rz", "sx", "x"], True, []),
        (0, ["cz", "rz", "sx", "x"], True, []),
        (1, ["cz", "rz", "sx", "x"], True, []),
        (2, ["cz", "rz", "sx", "x"], True, [True]),
        (3, ["cz", "rz", "sx", "x"], True, [True]),
        (2, ["cz", "rz", "sx", "x"], False, [True]),
        (2, None, False, [True]),
        # A native swap costs one two-qubit gate, so there is nothing to gain.
        (2, ["cz", "rz", "sx", "x", "swap"], True, []),
        (3, ["cz", "rz", "sx", "x", "swap"], False, []),
        (2, ["rzz", "cz", "rz", "sx", "x"], True, [False]),
        (3, ["rzz", "cz", "rz", "sx", "x"], False, [False]),
    )
    @unpack
    def test_translation_passmanager(self, level, basis, with_target, move_diagonal):
        """``generate_translation_passmanager`` adds the pass at level 2 or 3 without a native
        ``swap``, and with ``move_diagonal=False`` for a native ``rzz``."""
        target = None
        if with_target:
            target = Target.from_configuration(
                basis, num_qubits=2, coupling_map=CouplingMap.from_line(2)
            )
            basis = None
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        passes = []
        pm = common.generate_translation_passmanager(
            target, basis_gates=basis, optimization_level=level
        )
        pm.run(qc, callback=lambda **kwargs: passes.append(kwargs["pass_"]))
        found = [pass_.move_diagonal for pass_ in passes if isinstance(pass_, _AbsorbIntoSwaps)]
        self.assertEqual(found, move_diagonal)

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
