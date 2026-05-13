from tongsim_lite_protobuf import common_pb2 as _common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ConsoleCommand(_message.Message):
    __slots__ = ("console_command",)
    CONSOLE_COMMAND_FIELD_NUMBER: _ClassVar[int]
    console_command: str
    def __init__(self, console_command: _Optional[str] = ...) -> None: ...
