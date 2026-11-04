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
from qiskit.circuit.library import CU1Gate, QFTGate, qaoa_ansatz
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.transpiler import CouplingMap, PassManagerConfig, Target, generate_preset_pass_manager
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager
from qiskit.transpiler.preset_passmanagers import _swap_absorption, common
from qiskit.transpiler.preset_passmanagers._swap_absorption import _AbsorbIntoSwaps
from test import QiskitTestCase


def _circuit(num_qubits, gates, num_clbits=0, physical=True):
    """A circuit built from ``(method, *args)`` rows, on physical qubits unless
    ``physical=False``."""
    qc = QuantumCircuit(num_qubits, num_clbits)
    for method, *args in gates:
        getattr(qc, method)(*args)
    if physical:
        qc.ensure_physical()
    return qc


def _two_qubit_count(qc):
    return sum(inst.operation.num_qubits == 2 for inst in qc)


def _custom_gate(name, num_qubits, gates):
    """A custom gate with a standard gate's name, defined by ``gates``."""
    gate = Gate(name, num_qubits, [])
    gate.definition = _circuit(num_qubits, gates, physical=False)
    return gate


_BASIS = ["cz", "rz", "sx", "x"]
_CLIFFORD_BASIS = ["h", "s", "sdg", "x", "cx", "cz"]


def _line_target(num_qubits, basis=_BASIS):
    return Target.from_configuration(
        basis, num_qubits=num_qubits, coupling_map=CouplingMap.from_line(num_qubits)
    )


# Two-qubit gates after the SWAP on both of its qubits, so that it is not at the end of a wire.
_TAIL = [("cx", 0, 2), ("cx", 1, 2)]
_DIAG = np.diag([1, 1j, -1, -1j])
_NEAR_DIAG = _DIAG + np.diag([1e-13, 0, 0], 1)  # one off-diagonal entry of 1e-13
_CUSTOM_RZ = _custom_gate("rz", 1, [("h", 0)])
_CUSTOM_SWAP = _custom_gate("swap", 2, [("cx", 0, 1)])
_RZ_BODY = _circuit(1, [("rz", 0.2, 0)], num_clbits=1, physical=False)


@ddt
class TestAbsorbIntoSwaps(QiskitTestCase):
    """Test the ``_AbsorbIntoSwaps`` pass on circuits of at most 4 qubits."""

    @data(
        ("rxx", True, [("swap", 0, 1), ("rx", 0.3, 0), ("rxx", 0.5, 0, 1), *_TAIL]),
        ("ryy", True, [("swap", 0, 1), ("y", 0), ("ryy", 0.2, 1, 2), ("ryy", 0.5, 0, 1), *_TAIL]),
        ("rxx", False, [("rxx", 0.3, 0, 1), ("rx", 0.2, 0), ("swap", 0, 1), *_TAIL]),
        ("crx", True, [("swap", 0, 1), ("rz", 0.2, 0), ("x", 1), ("crx", 0.5, 0, 1), *_TAIL]),
        ("crx", False, [("crx", 0.5, 1, 0), ("sx", 0), ("t", 1), ("swap", 0, 1), *_TAIL]),
        ("csx", True, [("swap", 0, 1), ("s", 1), ("rx", 0.4, 0), ("csx", 1, 0), *_TAIL]),
        ("csx", False, [("csx", 0, 1), ("z", 0), ("swap", 0, 1), *_TAIL]),
        ("rzx", True, [("swap", 0, 1), ("p", 0.3, 0), ("sx", 1), ("rzx", 0.5, 0, 1), *_TAIL]),
        ("rzx", False, [("rzx", 0.5, 1, 0), ("rx", 0.2, 0), ("swap", 0, 1), *_TAIL]),
        # A barrier after the SWAP counts as a multi-qubit operation on both of its qubits.
        ("rxx", False, [("rxx", 0.3, 0, 1), ("x", 0), ("swap", 0, 1), ("cx", 0, 2), ("barrier",)]),
    )
    @unpack
    def test_gate_moves_next_to_swap(self, moved, after, gates):
        """A gate on the SWAP's qubits moves through operations it commutes with."""
        qc = _circuit(3, gates)

        out = _AbsorbIntoSwaps()(qc)

        dag = circuit_to_dag(out)
        (swap,) = dag.named_nodes("swap")
        neighbors = dag.op_successors(swap) if after else dag.op_predecessors(swap)
        self.assertEqual({node.name for node in neighbors}, {moved})
        self.assertEqual(dict(out.count_ops()), dict(qc.count_ops()))
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(
        ([("swap", 0, 1), ("rz", 0.2, 0), ("rzz", 0.4, 0, 1), *_TAIL], {"cx": 5, "p": 1, "rz": 1}),
        ([("cp", 0.3, 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL], {"cx": 5, "p": 3, "rz": 1}),
        ([("swap", 0, 1), ("rz", 0.2, 0), ("cz", 1, 0), *_TAIL], {"cx": 5, "p": 3, "rz": 1}),
        ([("swap", 0, 1), ("rz", 0.2, 0), ("crz", 0.4, 1, 0), *_TAIL], {"cx": 5, "p": 2, "rz": 1}),
        ([("crz", 0.4, 1, 0), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL], {"cx": 5, "p": 2, "rz": 1}),
        ([("swap", 0, 1), ("t", 0), ("cs", 0, 1), *_TAIL], {"cx": 5, "p": 3, "t": 1}),
        ([("csdg", 1, 0), ("tdg", 1), ("swap", 0, 1), *_TAIL], {"cx": 5, "p": 3, "tdg": 1}),
        (
            [("swap", 0, 1), ("z", 1), ("append", CU1Gate(0.6), [1, 0]), *_TAIL],
            {"cx": 5, "p": 3, "z": 1},
        ),
        # Without a supported two-qubit gate, the unitary alone must keep the pass running.
        (
            [("swap", 0, 1), ("s", 0), ("unitary", _DIAG, [0, 1]), ("barrier",)],
            {"cx": 3, "p": 2, "s": 1, "barrier": 1},
        ),
        ([("unitary", _DIAG, [1, 0]), ("s", 0), ("swap", 0, 1), *_TAIL], {"cx": 5, "p": 2, "s": 1}),
        (
            [("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1), ("rz", 0.7, 1), ("cx", 0, 1), *_TAIL],
            {"cx": 5, "p": 1, "rz": 1},
        ),
        (
            [("cx", 0, 1), ("rz", 0.7, 1), ("cx", 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL],
            {"cx": 5, "p": 1, "rz": 1},
        ),
        ([("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1), *_TAIL], {"cx": 4, "rz": 1}),
        ([("cx", 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), *_TAIL], {"cx": 4, "rz": 1}),
        ([("swap", 0, 1), ("sx", 0), ("cx", 1, 0), *_TAIL], {"cx": 4, "sx": 1}),
    )
    @unpack
    def test_fused_forms(self, gates, counts):
        """A SWAP and a diagonal gate or a CX are written as an exact short circuit."""
        qc = _circuit(3, gates)

        out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(dict(out.count_ops()), counts)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

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

        out = _AbsorbIntoSwaps()(qc)

        self.assertNotIn("swap", out.count_ops())
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_fused_form_keeps_small_phases(self):
        """The phase gates of a fused form are kept however small their angles are."""
        qc = _circuit(3, [("swap", 0, 1), ("z", 0), ("cp", 1e-12, 0, 1), *_TAIL])

        out = _AbsorbIntoSwaps()(qc)

        angles = sorted(abs(inst.operation.params[0]) for inst in out if inst.name == "p")
        np.testing.assert_allclose(angles, [5e-13, 5e-13, 5e-13], rtol=0, atol=1e-15)
        np.testing.assert_allclose(Operator(out).data, Operator(qc).data, rtol=0, atol=1e-14)

    def test_equal_units_share_a_block(self):
        """Equal units are written once and reused, and each use adds the block's global phase."""
        gates = [("cx", 0, 2), ("cx", 1, 4), ("cx", 3, 5)]
        for a, b in [(0, 1), (2, 3), (4, 5)]:
            gates[:0] = [("swap", a, b), ("rz", 0.2, a), ("rzz", 0.4, a, b)]
        qc = _circuit(6, gates)

        with patch.object(
            _swap_absorption, "_block_operations", wraps=_swap_absorption._block_operations
        ) as block_operations:
            out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(block_operations.call_count, 1)
        self.assertEqual(dict(out.count_ops()), {"cx": 12, "p": 3, "rz": 3})
        # Each block has the global phase -0.2 of rzz(0.4).
        self.assertAlmostEqual(out.global_phase, 2 * np.pi - 0.6)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(
        ([("swap", 0, 1), ("rz", 0.2, 0), ("rzz", 0.4, 0, 1), *_TAIL], None),
        ([("cx", 0, 1), ("rz", 0.7, 1), ("cx", 0, 1), ("s", 0), ("swap", 0, 1), *_TAIL], None),
        ([("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1), *_TAIL], {"cx": 4, "rz": 1}),
        # The rzz after the SWAP is passed over, and the CX before it is moved.
        (
            [("cx", 0, 1), ("rz", 0.2, 0), ("swap", 0, 1), ("rz", 0.3, 0), ("rzz", 0.4, 0, 1)]
            + _TAIL,
            {"cx": 4, "rz": 2, "rzz": 1},
        ),
    )
    @unpack
    def test_move_diagonal_false(self, gates, counts):
        """With ``move_diagonal=False`` a diagonal unit stays where it is and a CX still moves."""
        qc = _circuit(3, gates)

        out = _AbsorbIntoSwaps(move_diagonal=False)(qc)

        self.assertNotEqual(_AbsorbIntoSwaps()(qc), qc)
        if counts is None:
            self.assertEqual(out, qc)
        else:
            self.assertEqual(dict(out.count_ops()), counts)
            self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(
        ("h", [("swap", 0, 1), ("h", 0), ("cp", 0.3, 0, 1), *_TAIL]),
        ("barrier", [("swap", 0, 1), ("rz", 0.2, 0), ("barrier", 0), ("cp", 0.3, 0, 1), *_TAIL]),
        ("barrier", [("swap", 0, 1), ("rz", 0.2, 0), ("barrier", 0, 1), ("cp", 0.3, 0, 1), *_TAIL]),
        # An approximate commutation check would accept this angle; the pass must not.
        ("small ry", [("swap", 0, 1), ("ry", 1e-6, 0), ("cp", 0.3, 0, 1), *_TAIL]),
        ("rz on cx target", [("swap", 0, 1), ("rz", 0.2, 1), ("cx", 0, 1), *_TAIL]),
        ("cx; sx; cx", [("swap", 0, 1), ("s", 1), ("cx", 0, 1), ("sx", 1), ("cx", 0, 1), *_TAIL]),
        (
            "cx; D; cx with a gate on the control",
            [("swap", 0, 1), ("rz", 0.2, 0), ("p", 0.1, 1), ("cx", 0, 1), ("rz", 0.3, 1)]
            + [("rz", 0.4, 0), ("cx", 0, 1), *_TAIL],
        ),
        (
            "control flow",
            [("swap", 0, 1), ("if_test", (0, 1), _RZ_BODY, [0], [0]), ("cp", 0.3, 0, 1), *_TAIL],
        ),
        ("custom rz", [("swap", 0, 1), ("append", _CUSTOM_RZ, [0]), ("cp", 0.3, 0, 1), *_TAIL]),
        (
            "custom swap",
            [("append", _CUSTOM_SWAP, [0, 1]), ("rz", 0.2, 0), ("cp", 0.3, 0, 1), *_TAIL],
        ),
        # The pass only moves exactly diagonal unitaries.
        (
            "nearly diagonal unitary",
            [("swap", 0, 1), ("unitary", _NEAR_DIAG, [0, 2]), ("cp", 0.3, 0, 1), *_TAIL],
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

        self.assertEqual(_AbsorbIntoSwaps()(qc), qc, name)

    def test_missing_table_entry_blocks_move(self):
        """Without a stored commutation relation, nothing moves."""
        qc = _circuit(3, [("swap", 0, 1), ("rz", 0.2, 0), ("cp", 0.3, 0, 1), *_TAIL])
        with patch.object(
            _swap_absorption.SessionCommutationChecker,
            "check_commutation_entries",
            return_value=None,
        ):
            self.assertEqual(_AbsorbIntoSwaps()(qc), qc)

    @data(
        # A gate that both of two SWAPs on its pair could take moves to only one of them.
        (
            [("swap", 0, 1), ("cz", 0, 2), ("cp", 0.3, 0, 1), ("cz", 1, 2), ("swap", 0, 1), *_TAIL],
            [{"cx": 5, "p": 3, "cz": 2, "swap": 1}],
        ),
        # With a movable gate on each side of a SWAP, one of them is merged with it.
        (
            [("cp", 0.3, 0, 1), ("cz", 0, 2), ("swap", 0, 1), ("rz", 0.2, 0), ("cz", 1, 2)]
            + [("rzz", 0.4, 0, 1), *_TAIL],
            [
                {"cp": 1, "cz": 2, "rz": 1, "cx": 5, "p": 1},
                {"rzz": 1, "cz": 2, "rz": 1, "cx": 5, "p": 3},
            ],
        ),
        # At two SWAPs that share a qubit, the ``cp`` moves back to the first SWAP and the ``rzz``
        # forward to the second, each across the other.
        (
            [("h", 0), ("h", 1), ("h", 2), ("h", 3), ("swap", 0, 1), ("rz", 0.13, 1)]
            + [("rzz", 0.41, 1, 2), ("rz", 0.29, 1), ("cp", -0.37, 0, 1), ("swap", 1, 2)]
            + [("cx", 2, 3), ("cx", 0, 1)],
            [{"h": 4, "cx": 8, "p": 4, "rz": 2}],
        ),
    )
    @unpack
    def test_two_swaps(self, gates, counts):
        """Each gate is merged with at most one SWAP."""
        qc = _circuit(4, gates)

        out = _AbsorbIntoSwaps()(qc)

        self.assertIn(dict(out.count_ops()), counts)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_parameterized_gate(self):
        """A parameterized ``cp`` moves and keeps its parameter."""
        theta = Parameter("theta")
        qc = _circuit(3, [("swap", 0, 1), ("rz", 0.2, 0), ("cp", theta, 0, 1), *_TAIL])

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
                a, b = rng.sample(range(num_qubits), 2)
                angle = rng.uniform(-1, 1)
                name = rng.choice(
                    ("rz", "sx", "h", "s", "x", "cx", "cz", "cp", "rzz", "rxx", "swap")
                )
                if name == "rz":
                    qc.rz(angle, a)
                elif name in ("sx", "h", "s", "x"):
                    getattr(qc, name)(a)
                elif name in ("cp", "rzz", "rxx"):
                    getattr(qc, name)(angle, a, b)
                elif name == "cz" and rng.random() < 0.5:
                    qc.cx(a, b)
                    qc.rz(angle, b)
                    qc.cx(a, b)
                else:
                    getattr(qc, name)(a, b)
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)
            qc.ensure_physical()

            out = _AbsorbIntoSwaps()(qc)

            changed += out != qc
            with self.subTest(index=index):
                self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))
        self.assertGreater(changed, 0)


@ddt
class TestAbsorbIntoSwapsPresets(QiskitTestCase):
    """Test the pass in the preset pass managers, on circuits of at most 6 qubits."""

    @idata(itertools.product([None, 0, 1, 2, 3], ["default", "translator", "synthesis"]))
    @unpack
    def test_translation_plugins(self, level, method):
        """The built-in translation plugins add the pass at levels 2 and 3 only."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        passes = []
        pm = PassManagerStagePluginManager().get_passmanager_stage(
            "translation",
            method,
            PassManagerConfig(target=_line_target(2)),
            optimization_level=level,
        )
        pm.run(qc, callback=lambda **kwargs: passes.append(type(kwargs["pass_"]).__name__))
        self.assertEqual("_AbsorbIntoSwaps" in passes, level in (2, 3))

    @data(
        (None, _BASIS, True, []),
        (0, _BASIS, True, []),
        (1, _BASIS, True, []),
        (2, _BASIS, True, [True]),
        (3, _BASIS, True, [True]),
        (2, _BASIS, False, [True]),
        # Without basis gates or with a native swap, a swap costs one two-qubit gate.
        (2, None, False, []),
        (2, [*_BASIS, "swap"], True, []),
        (3, [*_BASIS, "swap"], False, []),
        (2, [*_BASIS, "rzz"], True, [False]),
        (3, [*_BASIS, "rzz"], False, [False]),
        # Diagonal gates are written with ``cx`` and ``p``, which these bases cannot express.
        (2, ["cx", "cp", "z"], True, [False]),
        (3, _CLIFFORD_BASIS, False, [False]),
    )
    @unpack
    def test_translation_passmanager(self, level, basis, with_target, move_diagonal):
        """``generate_translation_passmanager`` adds the pass at level 2 or 3 without a native
        ``swap``, and with ``move_diagonal=False`` for a native ``rzz`` or without ``p``."""
        target = _line_target(2, basis) if with_target else None
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        passes = []
        pm = common.generate_translation_passmanager(
            target, basis_gates=None if with_target else basis, optimization_level=level
        )
        pm.run(qc, callback=lambda **kwargs: passes.append(kwargs["pass_"]))
        found = [pass_.move_diagonal for pass_ in passes if isinstance(pass_, _AbsorbIntoSwaps)]
        self.assertEqual(found, move_diagonal)

    @idata(
        itertools.product(
            [2, 3],
            [
                (["cx", "cp", "z"], [("swap", 0, 1), ("z", 0), ("cp", 0.3, 0, 1), *_TAIL]),
                (_CLIFFORD_BASIS, [("swap", 0, 1), ("z", 0), ("cz", 0, 1), *_TAIL]),
            ],
        )
    )
    @unpack
    def test_target_without_phase_gates(self, level, case):
        """On a target that cannot run ``p`` gates, the circuit still compiles, with no more
        two-qubit gates than without the pass."""
        basis, gates = case
        qc = _circuit(3, gates, physical=False)
        options = {"target": Target.from_configuration(basis, num_qubits=3), "seed_transpiler": 1}

        out = generate_preset_pass_manager(level, routing_method="none", **options).run(qc)
        with patch.object(_AbsorbIntoSwaps, "run", lambda self, dag: dag):
            control = generate_preset_pass_manager(level, routing_method="none", **options).run(qc)

        self.assertLessEqual(_two_qubit_count(out), _two_qubit_count(control))
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data((2, {}), (3, {"initial_layout": [4, 3, 2, 1, 0]}))
    @unpack
    def test_qaoa_on_line(self, level, options):
        """Routed QAOA on a line needs fewer two-qubit gates than without the pass, and the
        pass runs after the layout is chosen, so the layout is the same."""
        cost = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, j], 1.0) for i in range(5) for j in range(i + 1, 5)], num_qubits=5
        )
        qc = qaoa_ansatz(cost, reps=1).assign_parameters([0.4, 0.9])
        target = _line_target(5)
        options = {"target": target, "seed_transpiler": 7, **options}

        out = generate_preset_pass_manager(level, **options).run(qc)
        with patch.object(_AbsorbIntoSwaps, "run", lambda self, dag: dag):
            control = generate_preset_pass_manager(level, **options).run(qc)

        self.assertLess(out.count_ops()["cz"], control.count_ops()["cz"])
        self.assertEqual(out.layout, control.layout)
        check_map = CheckMap(target)
        check_map(out)
        self.assertTrue(check_map.property_set["is_swap_mapped"])
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @idata(
        itertools.product(
            [2, 3],
            [
                {"target": _line_target(6)},
                {"basis_gates": _BASIS, "coupling_map": CouplingMap.from_line(6)},
                {"basis_gates": _BASIS},
            ],
        )
    )
    @unpack
    def test_qft(self, level, options):
        """A QFT keeps its operator, global phase included, with a target or ``basis_gates``."""
        qc = QuantumCircuit(6)
        qc.append(QFTGate(6), range(6))
        passes = []
        pm = generate_preset_pass_manager(level, seed_transpiler=1, **options)

        out = pm.run(qc, callback=lambda **kwargs: passes.append(type(kwargs["pass_"]).__name__))

        self.assertIn("_AbsorbIntoSwaps", passes)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))
