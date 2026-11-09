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
from test import QiskitTestCase

import numpy as np
from ddt import ddt, data, idata, unpack

from qiskit.circuit import Gate, IfElseOp, Parameter, QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import CPhaseGate, CU1Gate, CXGate, CZGate, PhaseGate, QFTGate, RXGate
from qiskit.circuit.library import ECRGate, RXXGate, RZGate, SwapGate, SXGate, XGate, ZGate
from qiskit.circuit.library import qaoa_ansatz
from qiskit.converters import circuit_to_dag
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.passmanager import BaseController
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler import CouplingMap, PassManagerConfig, Target, generate_preset_pass_manager
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes import CheckGateDirection, CheckMap, HLSConfig
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from qiskit.transpiler.preset_passmanagers import _swap_absorption
from qiskit.transpiler.preset_passmanagers._swap_absorption import _AbsorbIntoSwaps


def _circuit(num_qubits, gates, num_clbits=0, physical=True):
    """A circuit from ``(method, *args)`` rows, on physical qubits by default."""
    qc = QuantumCircuit(num_qubits, num_clbits)
    for method, *args in gates:
        getattr(qc, method)(*args)
    if physical:
        qc.ensure_physical()
    return qc


def _custom_gate(name, num_qubits, gates):
    """Return a gate named ``name`` with the given definition."""
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
_BUILTIN = ["default", "translator", "synthesis"]
_OPT_IN = ["absorb_swaps", "absorb_swaps_synthesis"]


def _pass_types(task, in_loop=False):
    """Yield ``(type name, in a do-while loop)`` for every pass in ``task``, recursively."""
    if isinstance(task, BaseController):
        for child in task.tasks:
            yield from _pass_types(child, in_loop or isinstance(task, DoWhileController))
    else:
        yield type(task).__name__, in_loop


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
        # Without a supported two-qubit gate, the unitary alone must keep the pass running.
        ("unitary", True, [("swap", 0, 1), ("s", 0), ("unitary", _DIAG, [0, 1]), ("barrier",)]),
        ("unitary", False, [("unitary", _DIAG, [1, 0]), ("s", 0), ("swap", 0, 1), *_TAIL]),
        ("cp", True, [("swap", 0, 1), ("rz", 0.2, 0), ("cp", Parameter("theta"), 0, 1), *_TAIL]),
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
        if qc.parameters:
            self.assertEqual(out.parameters, qc.parameters)
            out = out.assign_parameters([0.37])
            qc = qc.assign_parameters([0.37])
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
        """A SWAP and a diagonal standard gate or a CX are fused exactly."""
        qc = _circuit(3, gates)

        out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(dict(out.count_ops()), counts)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(
        # Different angle.
        [("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1), ("rz", 0.7, 1), ("cx", 0, 1)]
        + [("swap", 2, 3), ("rz", 0.2, 2), ("cx", 2, 3), ("rz", 0.3, 3), ("cx", 2, 3)],
        # Different gate.
        [("swap", 0, 1), ("rz", 0.2, 0), ("crz", 0.7, 0, 1)]
        + [("swap", 2, 3), ("rz", 0.2, 2), ("cp", 0.7, 2, 3)],
        # Reversed qargs.
        [("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1)]
        + [("swap", 2, 3), ("rz", 0.2, 3), ("cx", 3, 2)],
        # Opposite direction.
        [("swap", 0, 1), ("rz", 0.2, 0), ("cx", 0, 1)]
        + [("cx", 2, 3), ("rz", 0.2, 2), ("swap", 2, 3)],
    )
    def test_distinct_block_keys(self, gates):
        """Gates that differ only in angle, gate, qubit order or direction are fused exactly."""
        qc = _circuit(4, gates + [("cx", 0, 2), ("cx", 1, 3)])

        out = _AbsorbIntoSwaps()(qc)

        self.assertNotIn("swap", out.count_ops())
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    def test_fused_form_keeps_small_phases(self):
        """The phase gates of a fused form are kept however small their angles are."""
        qc = _circuit(3, [("swap", 0, 1), ("z", 0), ("cp", 1e-16, 0, 1), *_TAIL])

        out = _AbsorbIntoSwaps()(qc)

        angles = sorted(inst.operation.params[0] for inst in out if inst.name == "p")
        self.assertEqual(angles, [-5e-17, 5e-17, 5e-17])
        np.testing.assert_allclose(Operator(out).data, Operator(qc).data, rtol=0, atol=1e-14)

    def test_equal_blocks_are_reused(self):
        """Equal blocks are built once, and each use adds the block's global phase."""
        gates = 3 * [("swap", 0, 1), ("rz", 0.2, 0), ("rzz", 0.4, 0, 1), ("cx", 0, 2), ("cx", 1, 3)]
        qc = _circuit(4, gates)

        with patch.object(
            _swap_absorption, "_block_operations", wraps=_swap_absorption._block_operations
        ) as block_operations:
            out = _AbsorbIntoSwaps()(qc)

        self.assertEqual(block_operations.call_count, 1)
        self.assertEqual(dict(out.count_ops()), {"cx": 15, "p": 3, "rz": 3})
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
        """With ``move_diagonal=False`` a diagonal gate stays where it is and a CX still moves."""
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

    @data("qubits", "angle bounds")
    def test_check_without_passes(self, case):
        """With ``check=[]``, a move is not made when a block's phase gates are not supported."""
        target = Target(num_qubits=4)
        target.add_instruction(CXGate(), dict.fromkeys(itertools.permutations(range(4), 2)))

        def block(a, b):
            return [("swap", a, b), ("rz", 0.2, a), ("cp", 0.3, a, b), ("barrier",)]

        if case == "qubits":
            # The same block on (2, 3) first, where ``p`` is missing, then on (0, 1).
            target.add_instruction(PhaseGate(Parameter("a")), {(0,): None, (1,): None})
            qc = _circuit(4, block(2, 3) + block(0, 1))
        else:
            phases = {(q,): None for q in range(4)}
            target.add_instruction(PhaseGate(Parameter("a")), phases, angle_bounds=[(0, 2 * np.pi)])
            qc = _circuit(4, block(0, 1))

        self.assertEqual(_AbsorbIntoSwaps(target=target, check=[])(qc), qc)

    def test_check_errors(self):
        """A Qiskit error in the check rejects the moves; other errors propagate."""

        class Raise(AnalysisPass):
            """Raise ``error``."""

            def __init__(self, error):
                super().__init__()
                self.error = error

            def run(self, dag):
                raise self.error

        qc = _circuit(3, [("swap", 0, 1), ("z", 0), ("cx", 0, 1), *_TAIL])
        target = _line_target(3)
        with self.assertLogs(_swap_absorption.logger, "DEBUG"):
            out = _AbsorbIntoSwaps(target=target, check=[Raise(QiskitError("unsupported"))])(qc)
        self.assertEqual(out, qc)
        with self.assertRaises(ValueError):
            _AbsorbIntoSwaps(target=target, check=[Raise(ValueError("bug"))])(qc)

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

    @idata(itertools.product([None, 0, 1, 2, 3], _BUILTIN + _OPT_IN))
    @unpack
    def test_translation_plugins(self, level, method):
        """Only the opt-in translation plugins add the pass, at levels 2 and 3 only."""
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
        self.assertEqual("_AbsorbIntoSwaps" in passes, method in _OPT_IN and level in (2, 3))

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
        # With plugin options, the check would run the plugins more often.
        (2, _BASIS, True, [], {"unitary_synthesis_method": "sk"}),
        (2, _BASIS, True, [], {"unitary_synthesis_plugin_config": {}}),
        (2, _BASIS, True, [], {"hls_config": HLSConfig()}),
    )
    @unpack
    def test_translation_passmanager(self, level, basis, with_target, expected, options=None):
        """The opt-in translation stages add the pass by level, native gates and plugin options."""
        target = _line_target(2, basis) if with_target else None
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        basis_gates = None if with_target else basis
        config = PassManagerConfig(target=target, basis_gates=basis_gates, **(options or {}))
        for method in _BUILTIN + _OPT_IN:
            with self.subTest(method=method):
                passes = []
                pm = PassManagerStagePluginManager().get_passmanager_stage(
                    "translation", method, config, optimization_level=level
                )
                pm.run(qc, callback=lambda passes=passes, **kwargs: passes.append(kwargs["pass_"]))
                found = [p.move_diagonal for p in passes if isinstance(p, _AbsorbIntoSwaps)]
                self.assertEqual(found, expected if method in _OPT_IN else [])

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
        """A move is not made on a target without ``p`` gates."""
        basis, gates = case
        target = Target.from_configuration(basis, num_qubits=3)
        self._assert_moves_rejected(_circuit(3, gates), target, level)

    def _assert_moves_rejected(self, qc, target, level):
        """Assert that the check rejects the moves and the output equals that without the pass."""
        options = {"target": target, "seed_transpiler": 1, "routing_method": "none"}
        options["initial_layout"] = list(range(qc.num_qubits))
        options["translation_method"] = "absorb_swaps"
        checks = []
        supported = _AbsorbIntoSwaps._supported

        def spy(pass_, *args):
            checks.append(supported(pass_, *args))
            return checks[-1]

        with patch.object(_AbsorbIntoSwaps, "_supported", spy):
            out = generate_preset_pass_manager(level, **options).run(qc)
        with patch.object(_AbsorbIntoSwaps, "run", lambda self, dag: dag):
            control = generate_preset_pass_manager(level, **options).run(qc)
        self.assertEqual(checks, [False])
        self.assertEqual(out, control)

    @idata(itertools.product([2, 3], ["p", "rz"]))
    @unpack
    def test_single_qubit_gates_on_some_qubits(self, level, phase):
        """A move is not made when it needs single-qubit gates on qubits that lack them."""
        target = Target(num_qubits=3)
        pairs = dict.fromkeys(itertools.permutations(range(3), 2))
        if phase == "p":
            target.add_instruction(CXGate(), pairs)
            target.add_instruction(CPhaseGate(Parameter("a")), pairs)
            target.add_instruction(PhaseGate(Parameter("b")), {(2,): None})
            gates = [("swap", 0, 1), ("cp", 0.7, 0, 2), ("cp", 0.3, 0, 1), ("cp", 0.2, 1, 2)]
        else:
            target.add_instruction(CZGate(), pairs)
            target.add_instruction(SXGate(), {(q,): None for q in range(3)})
            target.add_instruction(XGate(), {(q,): None for q in range(3)})
            target.add_instruction(RZGate(Parameter("b")), {(0,): None})
            gates = [("cx", 1, 0), ("x", 0), ("swap", 0, 1), ("cz", 0, 2), ("cz", 1, 2)]
        self._assert_moves_rejected(_circuit(3, gates), target, level)

    @idata(itertools.product([2, 3], ["cp", "rxx"]))
    @unpack
    def test_fixed_parameters(self, level, gate):
        """A move is not made when the target has its gates only at other fixed angles."""
        target = Target(num_qubits=3)
        pairs = dict.fromkeys(itertools.permutations(range(3), 2))
        target.add_instruction(CXGate(), pairs)
        if gate == "cp":
            target.add_instruction(CPhaseGate(0.3), pairs)
            target.add_instruction(PhaseGate(1.0), {(q,): None for q in range(3)})
            gates = [("swap", 0, 1), ("cp", 0.3, 0, 2), ("cp", 0.3, 0, 1), ("cp", 0.3, 1, 2)]
        else:
            target.add_instruction(RXXGate(np.pi / 2), pairs)
            target.add_instruction(RXGate(0.123), {(q,): None for q in range(3)})
            target.add_instruction(RZGate(0.234), {(q,): None for q in range(3)})
            gates = [("swap", 0, 1), ("rxx", np.pi / 2, 0, 2), ("rxx", np.pi / 2, 0, 1), *_TAIL]
        self._assert_moves_rejected(_circuit(3, gates), target, level)

    @data(2, 3)
    def test_one_way_ecr_target(self, level):
        """A move on a one-way ECR target needs fewer two-qubit gates and keeps the direction."""
        target = Target(num_qubits=3)
        target.add_instruction(ECRGate(), {(0, 1): None, (1, 2): None})
        target.add_instruction(RZGate(Parameter("theta")), {(q,): None for q in range(3)})
        target.add_instruction(SXGate(), {(q,): None for q in range(3)})
        target.add_instruction(XGate(), {(q,): None for q in range(3)})
        config = PassManagerConfig(target=target, coupling_map=target.build_coupling_map())
        qc = _circuit(3, [("swap", 0, 1), ("z", 0), ("cx", 0, 1), ("cx", 1, 2), ("cx", 0, 1)])
        plugins = PassManagerStagePluginManager()

        method = "absorb_swaps"
        out = plugins.get_passmanager_stage("translation", method, config, level).run(qc)
        with patch.object(_AbsorbIntoSwaps, "run", lambda self, dag: dag):
            control = plugins.get_passmanager_stage("translation", method, config, level).run(qc)

        def two_qubit_gates(circuit):
            return sum(instruction.operation.num_qubits == 2 for instruction in circuit.data)

        self.assertLess(two_qubit_gates(out), two_qubit_gates(control))
        check_direction = CheckGateDirection(None, target=target)
        check_direction(out)
        self.assertTrue(check_direction.property_set["is_direction_mapped"])
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @data(2, 3)
    def test_custom_swap_and_one_way_cx(self, level):
        """A move is not made when the fused form needs a CX direction that the target lacks."""
        swap = _custom_gate("native_swap", 2, [("swap", 0, 1)])
        target = Target(num_qubits=3)
        target.add_instruction(CXGate(), {(0, 1): None, (0, 2): None, (1, 2): None})
        target.add_instruction(swap, {(0, 1): None})
        target.add_instruction(ZGate(), {(q,): None for q in range(3)})
        equivalence = QuantumCircuit(2)
        equivalence.append(swap, [0, 1])
        entry = SessionEquivalenceLibrary.get_entry(SwapGate())
        SessionEquivalenceLibrary.set_entry(SwapGate(), [equivalence])
        try:
            qc = _circuit(3, [("swap", 0, 1), ("z", 0), ("cx", 0, 1), *_TAIL])
            self._assert_moves_rejected(qc, target, level)
        finally:
            SessionEquivalenceLibrary.set_entry(SwapGate(), entry)

    def test_unitary_is_not_fused(self):
        """A diagonal ``UnitaryGate`` is moved with its matrix unchanged, not fused."""
        diagonal = np.diag([1 + 9e-13, 1, 1, 1])
        qc = _circuit(3, [("swap", 0, 1), ("z", 0), ("unitary", diagonal, [0, 1]), *_TAIL])

        out = _AbsorbIntoSwaps()(qc)

        self.assertNotEqual(out, qc)
        self.assertEqual(dict(out.count_ops()), dict(qc.count_ops()))
        np.testing.assert_array_equal(Operator(out).data, Operator(qc).data)

    @data((2, {}), (3, {"initial_layout": [4, 3, 2, 1, 0]}))
    @unpack
    def test_qaoa_on_line(self, level, options):
        """Routed QAOA on a line needs fewer two-qubit gates and keeps its layout."""
        cost = SparsePauliOp.from_sparse_list(
            [("ZZ", [i, j], 1.0) for i in range(5) for j in range(i + 1, 5)], num_qubits=5
        )
        qc = qaoa_ansatz(cost, reps=1).assign_parameters([0.4, 0.9])
        target = _line_target(5)
        options = {"target": target, "seed_transpiler": 7, **options}
        options["translation_method"] = "absorb_swaps"

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
        """A QFT absorbs routing SWAPs and keeps its operator, global phase included."""
        qc = QuantumCircuit(6)
        qc.append(QFTGate(6), range(6))
        counts = []
        has_routing = "target" in options or "coupling_map" in options
        options = {"seed_transpiler": 1, "translation_method": "absorb_swaps", **options}
        pm = generate_preset_pass_manager(level, **options)

        def callback(pass_, dag, **_):
            counts.append((type(pass_).__name__, dag.count_ops().get("swap", 0)))

        out = pm.run(qc, callback=callback)

        index = [name for name, _ in counts].index("_AbsorbIntoSwaps")
        # Without a coupling map there are no routing SWAPs to absorb.
        absorbed = counts[index][1] < counts[index - 1][1]
        self.assertEqual(absorbed, has_routing)
        self.assertEqual(Operator.from_circuit(out), Operator.from_circuit(qc))

    @idata(itertools.product([0, 1, 2, 3], [None, *_BUILTIN, *_OPT_IN]))
    @unpack
    def test_preset_pass_managers(self, level, method):
        """Only the opt-in plugins add the pass, to translation and to the optimization loop."""
        pm = generate_preset_pass_manager(level, target=_line_target(4), translation_method=method)
        passes = _pass_types(pm.to_flow_controller())
        found = {in_loop for name, in_loop in passes if name == "_AbsorbIntoSwaps"}
        self.assertEqual(found, {False, True} if method in _OPT_IN and level in (2, 3) else set())

    @idata(itertools.product([0, 1], zip(_OPT_IN, ["translator", "synthesis"])))
    @unpack
    def test_low_levels(self, level, methods):
        """At levels 0 and 1 the opt-in plugins give the output of the plugins they extend."""
        qc = QuantumCircuit(5)
        qc.append(QFTGate(5), range(5))
        outs = [
            generate_preset_pass_manager(
                level, target=_line_target(5), seed_transpiler=3, translation_method=method
            ).run(qc)
            for method in methods
        ]
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0].layout, outs[1].layout)

    def test_plugins_are_listed(self):
        """The opt-in plugins are registered translation plugins."""
        self.assertLessEqual({*_BUILTIN, *_OPT_IN}, set(list_stage_plugins("translation")))

    @idata(itertools.product([2, 3], _OPT_IN))
    @unpack
    def test_control_flow(self, level, method):
        """The opt-in plugins can be used on circuits with control flow."""
        target = _line_target(3, [*_BASIS, "measure"])
        target.add_instruction(IfElseOp, name="if_else")
        qc = QuantumCircuit(3, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.if_test((qc.clbits[0], 1), _RZ_BODY, [0], [0])
        qc.cp(0.3, 0, 2)
        qc.cx(1, 2)
        pm = generate_preset_pass_manager(
            level, target=target, seed_transpiler=1, translation_method=method
        )
        out = pm.run(qc)
        self.assertIn("if_else", out.count_ops())
