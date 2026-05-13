"""
entity.mixin
"""

from collections import defaultdict
from typing import ClassVar, TypeVar

from tongsim.connection.grpc.unary_api import UnaryAPI
from tongsim.connection.tags import ComponentType
from tongsim.core.world_context import WorldContext
from tongsim.logger import get_logger

from .entity import Entity

_logger = get_logger("entity")

T = TypeVar("T")


__all__ = [
    "AgentEntity",
    "CameraEntity",
]


async def _bind_ability_methods[T](entity: Entity, ability_type: type[T]) -> None:
    """
    将指定 Ability 的方法从 Impl 动态绑定到 Entity 实例上。

    该函数会将 Ability 协议中定义的所有公开方法，从 entity.as_(Ability) 实例中转发到 entity 本身.

    Args:
        entity (Entity): 要注入方法的 Entity 实例
        ability_type (type[Protocol]): Ability 协议类型

    """
    # TODO: 如何避免覆盖 Entity 自身已有的属性

    assert hasattr(
        ability_type, "__annotations__"
    ), "ability_type must be a Protocol type"

    impl = await entity.async_as_(ability_type)
    for attr in dir(ability_type):
        # 跳过私有方法和特殊方法
        if attr.startswith("_"):
            continue
        # 仅绑定 callable
        value = getattr(impl, attr, None)
        if not callable(value):
            continue

        setattr(entity, attr, value)


class MixinEntityBase(Entity):
    """
    用于通过 `_ability_types` 字段声明当前实体类型支持的能力（Ability Protocol 接口），
    并在创建时自动将这些能力方法绑定为实体成员方法。

    `_ability_types` 为类变量，需在子类中显式指定能力列表。
    """

    _ability_types: ClassVar[list[type]] = []

    @classmethod
    async def create(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        for ability in cls._ability_types:
            await _bind_ability_methods(self, ability)
        return self

    @classmethod
    async def from_grpc(
        cls, entity_id: str, world_context: WorldContext
    ) -> "MixinEntityBase":
        """
        通过 gRPC 查询构造 Entity。
        """

        # TODO: 为了保留 Entity Base 层的干净实现，此处出现了重复逻辑调用
        resp = await UnaryAPI.query_components(world_context.conn, entity_id)
        if resp is None:
            raise RuntimeError(f"Failed to query components for entity '{entity_id}'.")

        components: dict[ComponentType, list[str]] = defaultdict(list)
        for component_id, component_type in resp.items():
            components[component_type].append(component_id)

        _logger.debug(
            f"[Consturct mixin entity from gRPC] Entity {entity_id}  ---  ability-types: {list(cls._ability_types)}"
        )
        return await cls.create(entity_id, world_context, components)


class CameraEntity(MixinEntityBase):
    """
    TongSim 的相机实体
    """

    _ability_types: ClassVar[list[type]] = []


class AgentEntity(MixinEntityBase):
    """
    智能体
    """

    _ability_types: ClassVar[list[type]] = []
