# Adapted from huggingface/lerobot — Apache License 2.0
# Stripped to only what's needed for Feetech/STS3215 over serial.
# External deps: pyserial, scservo_sdk

from __future__ import annotations

import abc
import importlib.util
import logging
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, wraps
from pprint import pformat
from typing import TYPE_CHECKING, Protocol

logger = logging.getLogger(__name__)

type NameOrID = str | int
type Value = int | float

# ── Inline connection guards ───────────────────────────────────────────────────
# Replaces lerobot.utils.decorators without bringing in the whole package.

class DeviceNotConnectedError(RuntimeError): ...
class DeviceAlreadyConnectedError(RuntimeError): ...


def check_if_not_connected(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )
        return func(self, *args, **kwargs)
    return wrapper


def check_if_already_connected(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.__class__.__name__} is already connected.")
        return func(self, *args, **kwargs)
    return wrapper


# ── Optional serial import ────────────────────────────────────────────────────
_serial_available = importlib.util.find_spec("serial") is not None

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None


# ── Data types ─────────────────────────────────────────────────────────────────

class MotorNormMode(str, Enum):
    RANGE_0_100   = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES       = "degrees"


@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


@dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode
    motor_type_str: str | None = None
    recv_id: int | None = None


# ── SDK Protocol stubs ─────────────────────────────────────────────────────────
# These mirror the scservo_sdk interface so type checkers are happy.

class PortHandler(Protocol):
    is_open: bool
    baudrate: int
    packet_start_time: float
    packet_timeout: float
    tx_time_per_byte: float
    is_using: bool
    port_name: str
    ser: serial.Serial

    def __init__(self, port_name: str) -> None: ...
    def openPort(self): ...
    def closePort(self): ...
    def clearPort(self): ...
    def setPortName(self, port_name): ...
    def getPortName(self): ...
    def setBaudRate(self, baudrate): ...
    def getBaudRate(self): ...
    def getBytesAvailable(self): ...
    def readPort(self, length): ...
    def writePort(self, packet): ...
    def setPacketTimeout(self, packet_length): ...
    def setPacketTimeoutMillis(self, msec): ...
    def isPacketTimeout(self): ...
    def getCurrentTime(self): ...
    def getTimeSinceStart(self): ...
    def setupPort(self, cflag_baud): ...
    def getCFlagBaud(self, baudrate): ...


class PacketHandler(Protocol):
    def getTxRxResult(self, result): ...
    def getRxPacketError(self, error): ...
    def txPacket(self, port, txpacket): ...
    def rxPacket(self, port): ...
    def txRxPacket(self, port, txpacket): ...
    def ping(self, port, id): ...
    def action(self, port, id): ...
    def readTx(self, port, id, address, length): ...
    def readRx(self, port, id, length): ...
    def readTxRx(self, port, id, address, length): ...
    def read1ByteTx(self, port, id, address): ...
    def read1ByteRx(self, port, id): ...
    def read1ByteTxRx(self, port, id, address): ...
    def read2ByteTx(self, port, id, address): ...
    def read2ByteRx(self, port, id): ...
    def read2ByteTxRx(self, port, id, address): ...
    def read4ByteTx(self, port, id, address): ...
    def read4ByteRx(self, port, id): ...
    def read4ByteTxRx(self, port, id, address): ...
    def writeTxOnly(self, port, id, address, length, data): ...
    def writeTxRx(self, port, id, address, length, data): ...
    def write1ByteTxOnly(self, port, id, address, data): ...
    def write1ByteTxRx(self, port, id, address, data): ...
    def write2ByteTxOnly(self, port, id, address, data): ...
    def write2ByteTxRx(self, port, id, address, data): ...
    def write4ByteTxOnly(self, port, id, address, data): ...
    def write4ByteTxRx(self, port, id, address, data): ...
    def regWriteTxOnly(self, port, id, address, length, data): ...
    def regWriteTxRx(self, port, id, address, length, data): ...
    def syncReadTx(self, port, start_address, data_length, param, param_length): ...
    def syncWriteTxOnly(self, port, start_address, data_length, param, param_length): ...
    def broadcastPing(self, port): ...


class GroupSyncRead(Protocol):
    port: str
    ph: PortHandler
    start_address: int
    data_length: int
    last_result: bool
    is_param_changed: bool
    param: list
    data_dict: dict

    def __init__(self, port: PortHandler, ph: PacketHandler, start_address: int, data_length: int) -> None: ...
    def makeParam(self): ...
    def addParam(self, id): ...
    def removeParam(self, id): ...
    def clearParam(self): ...
    def txPacket(self): ...
    def rxPacket(self): ...
    def txRxPacket(self): ...
    def isAvailable(self, id, address, data_length): ...
    def getData(self, id, address, data_length): ...


class GroupSyncWrite(Protocol):
    port: str
    ph: PortHandler
    start_address: int
    data_length: int
    is_param_changed: bool
    param: list
    data_dict: dict

    def __init__(self, port: PortHandler, ph: PacketHandler, start_address: int, data_length: int) -> None: ...
    def makeParam(self): ...
    def addParam(self, id, data): ...
    def removeParam(self, id): ...
    def changeParam(self, id, data): ...
    def clearParam(self): ...
    def txPacket(self): ...


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_ctrl_table(model_ctrl_table: dict, model: str) -> dict:
    ctrl_table = model_ctrl_table.get(model)
    if ctrl_table is None:
        raise KeyError(f"Control table for {model=} not found.")
    return ctrl_table


def get_address(model_ctrl_table: dict, model: str, data_name: str) -> tuple[int, int]:
    ctrl_table = get_ctrl_table(model_ctrl_table, model)
    addr_bytes = ctrl_table.get(data_name)
    if addr_bytes is None:
        raise KeyError(f"Address for '{data_name}' not found in {model} control table.")
    return addr_bytes


def assert_same_address(model_ctrl_table: dict, motor_models: list[str], data_name: str) -> None:
    all_addr, all_bytes = [], []
    for model in motor_models:
        addr, nbytes = get_address(model_ctrl_table, model, data_name)
        all_addr.append(addr)
        all_bytes.append(nbytes)
    if len(set(all_addr)) != 1:
        raise NotImplementedError(f"Motors use different addresses for '{data_name}'.")
    if len(set(all_bytes)) != 1:
        raise NotImplementedError(f"Motors use different byte sizes for '{data_name}'.")


# ── Base classes ───────────────────────────────────────────────────────────────

class MotorsBusBase(abc.ABC):
    def __init__(self, port: str, motors: dict[str, Motor], calibration: dict[str, MotorCalibration] | None = None):
        self.port = port
        self.motors = motors
        self.calibration = calibration if calibration else {}

    @abc.abstractmethod
    def connect(self, handshake: bool = True) -> None: ...

    @abc.abstractmethod
    def disconnect(self, disable_torque: bool = True) -> None: ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool: ...

    @abc.abstractmethod
    def read(self, data_name: str, motor: str) -> Value: ...

    @abc.abstractmethod
    def write(self, data_name: str, motor: str, value: Value) -> None: ...

    @abc.abstractmethod
    def sync_read(self, data_name: str, motors: str | list[str] | None = None) -> dict[str, Value]: ...

    @abc.abstractmethod
    def sync_write(self, data_name: str, values: dict[str, Value]) -> None: ...

    @abc.abstractmethod
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None: ...

    @abc.abstractmethod
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None: ...

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]: ...

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None: ...


class SerialMotorsBus(MotorsBusBase):
    apply_drive_mode: bool
    available_baudrates: list[int]
    default_baudrate: int
    default_timeout: int
    model_baudrate_table: dict
    model_ctrl_table: dict
    model_encoding_table: dict
    model_number_table: dict[str, int]
    model_resolution_table: dict[str, int]
    normalized_data: list[str]

    def __init__(self, port: str, motors: dict[str, Motor], calibration: dict[str, MotorCalibration] | None = None):
        if not _serial_available:
            raise ImportError("pyserial is required. Run: pip install pyserial")
        super().__init__(port, motors, calibration)

        self.port_handler: PortHandler
        self.packet_handler: PacketHandler
        self.sync_reader: GroupSyncRead
        self.sync_writer: GroupSyncWrite
        self._comm_success: int
        self._no_error: int

        self._id_to_model_dict = {m.id: m.model for m in self.motors.values()}
        self._id_to_name_dict  = {m.id: motor  for motor, m in self.motors.items()}
        self._model_nb_to_model_dict = {v: k for k, v in self.model_number_table.items()}

        self._validate_motors()

    def __len__(self):
        return len(self.motors)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors: \n{pformat(self.motors, indent=8, sort_dicts=False)},\n"
            ")"
        )

    # We only ever use one model type per bus in these experiments, so this is always False.
    @cached_property
    def _has_different_ctrl_tables(self) -> bool:
        return False

    @cached_property
    def models(self) -> list[str]:
        return [m.model for m in self.motors.values()]

    @cached_property
    def ids(self) -> list[int]:
        return [m.id for m in self.motors.values()]

    def _model_nb_to_model(self, motor_nb: int) -> str:
        return self._model_nb_to_model_dict[motor_nb]

    def _id_to_model(self, motor_id: int) -> str:
        return self._id_to_model_dict[motor_id]

    def _id_to_name(self, motor_id: int) -> str:
        return self._id_to_name_dict[motor_id]

    def _get_motor_id(self, motor: NameOrID) -> int:
        if isinstance(motor, str):
            return self.motors[motor].id
        elif isinstance(motor, int):
            return motor
        raise TypeError(f"'{motor}' should be int or str.")

    def _get_motor_model(self, motor: NameOrID) -> str:
        if isinstance(motor, str):
            return self.motors[motor].model
        elif isinstance(motor, int):
            return self._id_to_model_dict[motor]
        raise TypeError(f"'{motor}' should be int or str.")

    def _get_motors_list(self, motors: NameOrID | Sequence[NameOrID] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        elif isinstance(motors, str):
            return [motors]
        elif isinstance(motors, int):
            return [self._id_to_name(motors)]
        elif isinstance(motors, Sequence):
            return [m if isinstance(m, str) else self._id_to_name(m) for m in motors]
        raise TypeError(motors)

    def _get_ids_values_dict(self, values: Value | dict[str, Value] | None) -> dict[int, Value]:
        if isinstance(values, (int, float)):
            return dict.fromkeys(self.ids, values)
        elif isinstance(values, dict):
            return {self.motors[motor].id: val for motor, val in values.items()}
        raise TypeError(f"'values' must be a single value or dict, got {values}")

    def _validate_motors(self) -> None:
        if len(self.ids) != len(set(self.ids)):
            raise ValueError(f"Duplicate motor IDs!\n{self}")
        for model in self.models:
            get_ctrl_table(self.model_ctrl_table, model)

    def _is_comm_success(self, comm: int) -> bool:
        return comm == self._comm_success

    def _is_error(self, error: int) -> bool:
        return error != self._no_error

    def _assert_motors_exist(self) -> None:
        expected_models = {m.id: self.model_number_table[m.model] for m in self.motors.values()}
        found_models = {}
        for id_ in self.ids:
            model_nb = self.ping(id_)
            if model_nb is not None:
                found_models[id_] = model_nb

        missing_ids  = [id_ for id_ in self.ids if id_ not in found_models]
        wrong_models = {
            id_: (expected_models[id_], found_models[id_])
            for id_ in found_models
            if expected_models.get(id_) != found_models[id_]
        }

        if missing_ids or wrong_models:
            lines = [f"{self.__class__.__name__} motor check failed on port '{self.port}':"]
            if missing_ids:
                lines.append("\nMissing IDs:")
                lines.extend(f"  - {id_}" for id_ in missing_ids)
            if wrong_models:
                lines.append("\nWrong models:")
                lines.extend(
                    f"  - {id_}: expected {exp}, found {found}"
                    for id_, (exp, found) in wrong_models.items()
                )
            raise RuntimeError("\n".join(lines))

    @abc.abstractmethod
    def _assert_protocol_is_compatible(self, instruction_name: str) -> None: ...

    @property
    def is_connected(self) -> bool:
        return self.port_handler.is_open

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        self._connect(handshake)
        self.set_timeout()

    def _connect(self, handshake: bool = True) -> None:
        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
            elif handshake:
                self._handshake()
        except (FileNotFoundError, OSError, serial.SerialException) as e:
            raise ConnectionError(
                f"Could not connect on port '{self.port}'. Check the port and try again."
            ) from e

    @abc.abstractmethod
    def _handshake(self) -> None: ...

    @check_if_not_connected
    def disconnect(self, disable_torque: bool = True) -> None:
        if disable_torque:
            self.port_handler.clearPort()
            self.port_handler.is_using = False
            self.disable_torque(num_retry=5)
        self.port_handler.closePort()

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    def set_timeout(self, timeout_ms: int | None = None):
        timeout_ms = timeout_ms if timeout_ms is not None else self.default_timeout
        self.port_handler.setPacketTimeoutMillis(timeout_ms)

    def get_baudrate(self) -> int:
        return self.port_handler.getBaudRate()

    def set_baudrate(self, baudrate: int) -> None:
        if self.port_handler.getBaudRate() != baudrate:
            self.port_handler.setBaudRate(baudrate)
            if self.port_handler.getBaudRate() != baudrate:
                raise RuntimeError("Failed to set baud rate.")

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool: ...

    @abc.abstractmethod
    def read_calibration(self) -> dict[str, MotorCalibration]: ...

    @abc.abstractmethod
    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None: ...

    def reset_calibration(self, motors: NameOrID | Sequence[NameOrID] | None = None) -> None:
        for motor in self._get_motors_list(motors):
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            self.write("Homing_Offset", motor, 0, normalize=False)
            self.write("Min_Position_Limit", motor, 0, normalize=False)
            self.write("Max_Position_Limit", motor, max_res, normalize=False)
        self.calibration = {}

    @abc.abstractmethod
    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]: ...

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for '{motor}': min == max.")

            bounded = min(max_, max(min_, val))
            mode = self.motors[motor].norm_mode
            if mode is MotorNormMode.RANGE_M100_100:
                norm = (((bounded - min_) / (max_ - min_)) * 200) - 100
                normalized[id_] = -norm if drive_mode else norm
            elif mode is MotorNormMode.RANGE_0_100:
                norm = ((bounded - min_) / (max_ - min_)) * 100
                normalized[id_] = 100 - norm if drive_mode else norm
            elif mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                normalized[id_] = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError
        return normalized

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for '{motor}': min == max.")

            mode = self.motors[motor].norm_mode
            if mode is MotorNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                unnormalized[id_] = int(((min(100.0, max(-100.0, val)) + 100) / 200) * (max_ - min_) + min_)
            elif mode is MotorNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                unnormalized[id_] = int((min(100.0, max(0.0, val)) / 100) * (max_ - min_) + min_)
            elif mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                unnormalized[id_] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError
        return unnormalized

    @abc.abstractmethod
    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]: ...

    @abc.abstractmethod
    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]: ...

    def _serialize_data(self, value: int, length: int) -> list[int]:
        if value < 0:
            raise ValueError(f"Negative values not allowed: {value}")
        max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(length)
        if max_value is None:
            raise NotImplementedError(f"Unsupported byte size: {length}")
        if value > max_value:
            raise ValueError(f"Value {value} exceeds max for {length} bytes ({max_value}).")
        return self._split_into_byte_chunks(value, length)

    @abc.abstractmethod
    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]: ...

    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        id_ = self._get_motor_id(motor)
        for n_try in range(1 + num_retry):
            model_number, comm, error = self.packet_handler.ping(self.port_handler, id_)
            if self._is_comm_success(comm):
                break
            logger.debug(f"ping failed {id_=}: {n_try=} {comm=} {error=}")

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            return None
        if self._is_error(error):
            if raise_on_error:
                raise RuntimeError(self.packet_handler.getRxPacketError(error))
            return None
        return model_number

    @abc.abstractmethod
    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None: ...

    @check_if_not_connected
    def read(self, data_name: str, motor: str, *, normalize: bool = True, num_retry: int = 0) -> Value:
        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)
        err_msg = f"Failed to read '{data_name}' on {id_=} after {num_retry + 1} tries."
        value, _, _ = self._read(addr, length, id_, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)
        decoded = self._decode_sign(data_name, {id_: value})
        if normalize and data_name in self.normalized_data:
            return self._normalize(decoded)[id_]
        return decoded[id_]

    def _read(self, address: int, length: int, motor_id: int, *, num_retry: int = 0,
              raise_on_error: bool = True, err_msg: str = "") -> tuple[int, int, int]:
        read_fn = {1: self.packet_handler.read1ByteTxRx,
                   2: self.packet_handler.read2ByteTxRx,
                   4: self.packet_handler.read4ByteTxRx}.get(length)
        if read_fn is None:
            raise ValueError(f"Unsupported length: {length}")

        for n_try in range(1 + num_retry):
            value, comm, error = read_fn(self.port_handler, motor_id, address)
            if self._is_comm_success(comm):
                break
            logger.debug(f"Failed read @{address=} ({length=}) on {motor_id=} ({n_try=})")

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        if self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")
        return value, comm, error

    @check_if_not_connected
    def write(self, data_name: str, motor: str, value: Value, *, normalize: bool = True, num_retry: int = 0) -> None:
        id_ = self.motors[motor].id
        model = self.motors[motor].model
        addr, length = get_address(self.model_ctrl_table, model, data_name)
        int_value = int(value)
        if normalize and data_name in self.normalized_data:
            int_value = self._unnormalize({id_: value})[id_]
        int_value = self._encode_sign(data_name, {id_: int_value})[id_]
        err_msg = f"Failed to write '{data_name}' on {id_=} with '{int_value}' after {num_retry + 1} tries."
        self._write(addr, length, id_, int_value, num_retry=num_retry, raise_on_error=True, err_msg=err_msg)

    def _write(self, addr: int, length: int, motor_id: int, value: int, *,
               num_retry: int = 0, raise_on_error: bool = True, err_msg: str = "") -> tuple[int, int]:
        data = self._serialize_data(value, length)
        for n_try in range(1 + num_retry):
            comm, error = self.packet_handler.writeTxRx(self.port_handler, motor_id, addr, length, data)
            if self._is_comm_success(comm):
                break
            logger.debug(f"Failed write @{addr=} on id={motor_id} with {value=} ({n_try=})")

        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        if self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")
        return comm, error

    @check_if_not_connected
    def sync_read(self, data_name: str, motors: NameOrID | Sequence[NameOrID] | None = None,
                  *, normalize: bool = True, num_retry: int = 0) -> dict[str, Value]:
        self._assert_protocol_is_compatible("sync_read")
        names = self._get_motors_list(motors)
        ids   = [self.motors[m].id    for m in names]
        model = next(iter(self.motors[m].model for m in names))
        addr, length = get_address(self.model_ctrl_table, model, data_name)
        err_msg = f"Failed to sync read '{data_name}' on {ids=} after {num_retry + 1} tries."
        raw_ids_values, _ = self._sync_read(addr, length, ids, num_retry=num_retry,
                                            raise_on_error=True, err_msg=err_msg)
        decoded = self._decode_sign(data_name, raw_ids_values)
        if normalize and data_name in self.normalized_data:
            normalized = self._normalize(decoded)
            return {self._id_to_name(id_): v for id_, v in normalized.items()}
        return {self._id_to_name(id_): v for id_, v in decoded.items()}

    def _sync_read(self, addr: int, length: int, motor_ids: list[int], *, num_retry: int = 0,
                   raise_on_error: bool = True, err_msg: str = "") -> tuple[dict[int, int], int]:
        self._setup_sync_reader(motor_ids, addr, length)
        for n_try in range(1 + num_retry):
            comm = self.sync_reader.txRxPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(f"Failed sync read @{addr=} on {motor_ids=} ({n_try=})")
        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        return {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}, comm

    def _setup_sync_reader(self, motor_ids: list[int], addr: int, length: int) -> None:
        self.sync_reader.clearParam()
        self.sync_reader.start_address = addr
        self.sync_reader.data_length   = length
        for id_ in motor_ids:
            self.sync_reader.addParam(id_)

    @check_if_not_connected
    def sync_write(self, data_name: str, values: Value | dict[str, Value],
                   *, normalize: bool = True, num_retry: int = 0) -> None:
        raw_ids_values = self._get_ids_values_dict(values)
        model = next(iter(self._id_to_model(id_) for id_ in raw_ids_values))
        addr, length = get_address(self.model_ctrl_table, model, data_name)
        int_ids_values = {id_: int(v) for id_, v in raw_ids_values.items()}
        if normalize and data_name in self.normalized_data:
            int_ids_values = self._unnormalize(raw_ids_values)
        int_ids_values = self._encode_sign(data_name, int_ids_values)
        err_msg = f"Failed sync write '{data_name}' with {int_ids_values} after {num_retry + 1} tries."
        self._sync_write(addr, length, int_ids_values, num_retry=num_retry,
                         raise_on_error=True, err_msg=err_msg)

    def _sync_write(self, addr: int, length: int, ids_values: dict[int, int],
                    num_retry: int = 0, raise_on_error: bool = True, err_msg: str = "") -> int:
        self._setup_sync_writer(ids_values, addr, length)
        for n_try in range(1 + num_retry):
            comm = self.sync_writer.txPacket()
            if self._is_comm_success(comm):
                break
            logger.debug(f"Failed sync write @{addr=} with {ids_values=} ({n_try=})")
        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        return comm

    def _setup_sync_writer(self, ids_values: dict[int, int], addr: int, length: int) -> None:
        self.sync_writer.clearParam()
        self.sync_writer.start_address = addr
        self.sync_writer.data_length   = length
        for id_, value in ids_values.items():
            self.sync_writer.addParam(id_, self._serialize_data(value, length))
