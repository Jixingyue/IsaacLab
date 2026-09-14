# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""子模块，提供对 Python 3.7+ ``dataclasses`` 模块的封装。"""

import inspect
import types
from collections.abc import Callable
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, field, replace
from typing import Any, ClassVar

from .dict import class_to_dict, update_class_from_dict

_CONFIGCLASS_METHODS = ["to_dict", "from_dict", "replace", "copy", "validate"]
"""在运行时添加到 dataclass 的类方法列表。"""

"""
dataclass 的封装。
"""


def __dataclass_transform__():
    """为 PyLance 添加注解装饰器。"""
    return lambda a: a


@__dataclass_transform__()
def configclass(cls, **kwargs):
    """对 `dataclass` 功能的封装，添加额外的检查和工具方法。

    从 Python 3.7 开始，标准 dataclass 存在两个主要问题，使其无法通用于配置场景：

    1. 要求所有成员都必须有类型注解。
    2. 要求显式使用 :meth:`field(default_factory=...)` 来重新初始化可变变量。

    本函数提供了一个装饰器，封装了 Python 的 `dataclass`_ 工具来处理上述两个问题。
    同时还提供了字典 <-> 类转换以及便捷复制类实例的辅助函数。

    用法：

    .. code-block:: python

        from dataclasses import MISSING

        from isaaclab.utils.configclass import configclass


        @configclass
        class ViewerCfg:
            eye: list = [7.5, 7.5, 7.5]  # 故意省略 field
            lookat: list = field(default_factory=[0.0, 0.0, 0.0])


        @configclass
        class EnvCfg:
            num_envs: int = MISSING
            episode_length: int = 2000
            viewer: ViewerCfg = ViewerCfg()


        # 创建配置实例
        env_cfg = EnvCfg(num_envs=24)

        # 以字典形式打印信息
        print(env_cfg.to_dict())

        # 创建配置的副本
        env_cfg_copy = env_cfg.copy()

        # 使用关键字参数替换指定字段
        env_cfg_copy = env_cfg_copy.replace(num_envs=32)

    Args:
        cls: 要封装的类。
        **kwargs: 传递给 :func:`dataclass` 的额外参数。

    Returns:
        封装后的类。

    .. _dataclass: https://docs.python.org/3/library/dataclasses.html
    """
    # 添加类型注解
    _add_annotation_types(cls)
    # 添加字段工厂
    _process_mutable_types(cls)
    # 复制可变成员
    # 注意：我们检查用户是否定义了 __post_init__ 函数，如果存在则将其与我们自己的函数组合
    if hasattr(cls, "__post_init__"):
        setattr(cls, "__post_init__", _combined_function(cls.__post_init__, _custom_post_init))
    else:
        setattr(cls, "__post_init__", _custom_post_init)
    # 添加字典转换的辅助函数
    setattr(cls, "to_dict", _class_to_dict)
    setattr(cls, "from_dict", _update_class_from_dict)
    setattr(cls, "replace", _replace_class_with_kwargs)
    setattr(cls, "copy", _copy_class)
    setattr(cls, "validate", _validate)
    # 封装 dataclass
    cls = dataclass(cls, **kwargs)
    # 返回封装后的类
    return cls


"""
字典 <-> 类操作。

这里重新定义以添加新的文档字符串。
"""


def _class_to_dict(obj: object) -> dict[str, Any]:
    """递归地将对象转换为字典。

    Args:
        obj: 要转换的对象。

    Returns:
        转换后的字典映射。
    """
    return class_to_dict(obj)


def _update_class_from_dict(obj, data: dict[str, Any]) -> None:
    """读取字典并递归地设置对象变量。

    该函数对类成员属性执行原地更新。

    Args:
        obj: 要更新的对象。
        data: 用于更新的输入（嵌套）字典。

    Raises:
        TypeError: 当输入不是字典时。
        ValueError: 当字典中的值与默认配置类型不匹配时。
        KeyError: 当字典中的键在默认配置类型中不存在时。
    """
    update_class_from_dict(obj, data, _ns="")


def _replace_class_with_kwargs(obj: object, **kwargs) -> object:
    """返回一个新对象，将指定字段替换为新值。

    这对于冻结类尤其有用。用法示例：

    .. code-block:: python

        @configclass(frozen=True)
        class C:
            x: int
            y: int


        c = C(1, 2)
        c1 = c.replace(x=3)
        assert c1.x == 3 and c1.y == 2

    Args:
        obj: 要替换的对象。
        **kwargs: 要替换的字段及其新值。

    Returns:
        新的对象。
    """
    return replace(obj, **kwargs)


def _copy_class(obj: object) -> object:
    """返回一个与原始对象具有相同字段的新对象。"""
    return replace(obj)


"""
私有辅助函数。
"""


def _add_annotation_types(cls):
    """为 dataclass 中的所有元素添加注解。

    在 Python 的定义中，字段被定义为具有类型注解的类变量。

    如果没有提供类型注解，dataclass 在调用 :func:`__dict__()` 时会忽略这些成员。
    本函数为类变量添加这些注解，以防止用户忘记指定类型注解时出现任何问题。

    这使得以下操作成为可行：

    @dataclass
    class State:
        pos = (0.0, 0.0, 0.0)
           ^^
           如果不使用本函数，将返回以下类型错误：
           TypeError: 'pos' is a field but has no type annotation
    """
    # 获取类型提示
    hints = {}
    # 遍历类继承链
    # 我们首先添加基类的注解
    for base in reversed(cls.__mro__):
        # 检查基类是否为 object
        if base is object:
            continue
        # 获取基类的注解
        ann = base.__dict__.get("__annotations__", {})
        # 直接添加基类的所有注解
        hints.update(ann)
        # 遍历基类的成员
        # 注意：不要将其改为 dir(base)，因为它会按字母顺序排列成员。
        #   这是不可取的，因为成员的顺序在某些情况下很重要。
        for key in base.__dict__:
            # 获取类成员
            value = getattr(base, key)
            # 跳过成员
            if _skippable_class_member(key, value, hints):
                continue
            # 为没有显式类型注解的成员添加类型注解
            # 对于这些成员，我们从默认值推导类型
            if not isinstance(value, type):
                if key not in hints:
                    # 检查变量类型是否不是 MISSING
                    # 我们无法从 MISSING 推导类型！
                    if value is MISSING:
                        raise TypeError(
                            f"Missing type annotation for '{key}' in class '{cls.__name__}'."
                            " Please add a type annotation or set a default value."
                        )
                    # 添加类型注解
                    hints[key] = type(value)
            elif key != value.__name__:
                # 注意：我们不想为嵌套的 configclass 添加类型注解。因此，我们检查
                #   类型的名称是否与变量的名称匹配。
                # 从 Python 3.10 开始，类型提示以字符串形式存储
                hints[key] = f"type[{value.__name__}]"

    # 注意：不要更改这一行。由于继承的原因，`cls.__dict__.get("__annotations__", {})` 与
    #   `cls.__annotations__` 不同。
    cls.__annotations__ = cls.__dict__.get("__annotations__", {})
    cls.__annotations__ = hints


def _validate(obj: object, prefix: str = "") -> list[str]:
    """检查 configclass 对象的有效性。

    该函数检查对象是否为有效的 configclass 对象。有效的 configclass 对象不包含 MISSING
    条目。

    Args:
        obj: 要检查的对象。
        prefix: 添加到缺失字段的前缀。默认为 ''。

    Returns:
        缺失字段的列表。

    Raises:
        TypeError: 当对象不是有效的配置对象时。
    """
    missing_fields = []

    if type(obj).__name__ == "MeshConverterCfg":
        return missing_fields

    if type(obj) is type(MISSING):
        missing_fields.append(prefix)
        return missing_fields
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            current_path = f"{prefix}[{index}]"
            missing_fields.extend(_validate(item, prefix=current_path))
        return missing_fields
    elif isinstance(obj, dict):
        # 将任何非字符串键转换为字符串，以允许对具有非字符串键的字典进行验证
        if any(not isinstance(key, str) for key in obj.keys()):
            obj_dict = {str(key): value for key, value in obj.items()}
        else:
            obj_dict = obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return missing_fields

    for key, value in obj_dict.items():
        # 忽略内置属性
        if key.startswith("__"):
            continue
        current_path = f"{prefix}.{key}" if prefix else key
        missing_fields.extend(_validate(value, prefix=current_path))

    # 仅在顶层调用时抛出错误
    if prefix == "" and missing_fields:
        formatted_message = "\n".join(f"  - {field}" for field in missing_fields)
        raise TypeError(
            f"Missing values detected in object {obj.__class__.__name__} for the following"
            f" fields:\n{formatted_message}\n"
        )
    return missing_fields


def _process_mutable_types(cls):
    """通过 :obj:`dataclasses.Field` 初始化所有可变元素，以避免不必要的报错。

    默认情况下，dataclass 要求使用 :obj:`field(default_factory=...)` 来在每次创建新的类实例时
    重新初始化可变对象。如果成员具有可变类型且创建时未指定 `field(default_factory=...)`，
    则 Python 会抛出错误，要求使用 `default_factory`。

    此外，Python 仅在类型为 list、set 或 dict 时显式检查字段规范。
    这遗漏了类型本身就是类的情况。因此，代码会悄悄携带一个可能导致不良效果的 bug。

    本函数处理了这个问题

    这使得以下操作成为可行：

    @dataclass
    class State:
        pos: list = [0.0, 0.0, 0.0]
           ^^
           如果不使用本函数，将返回以下值错误：
           ValueError: mutable default <class 'list'> for field pos is not allowed: use default_factory
    """
    # 注意：需要按照与注解相同的顺序设置。否则，
    #   会报缺少位置参数的错误。
    ann = cls.__dict__.get("__annotations__", {})

    # 遍历所有类成员并存储在字典中
    class_members = {}
    for base in reversed(cls.__mro__):
        # 检查基类是否为 object
        if base is object:
            continue
        # 遍历基类的成员
        for key in base.__dict__:
            # 获取类成员
            f = getattr(base, key)
            # 跳过成员
            if _skippable_class_member(key, f):
                continue
            # 存储类成员（如果它不是类型或已存在于注解中）
            if not isinstance(f, type) or key in ann:
                class_members[key] = f
        # 遍历基类的数据字段
        # 在上一次调用中，成为 dataclass 字段的内容已从类成员中移除
        # 因此我们需要在这里直接将它们作为 dataclass 字段添加回来
        for key, f in base.__dict__.get("__dataclass_fields__", {}).items():
            # 存储类成员
            if not isinstance(f, type):
                class_members[key] = f

    # 检查所有注解是否存在于类成员中
    # 注意：主要用于调试目的
    if len(class_members) != len(ann):
        raise ValueError(
            f"In class '{cls.__name__}', number of annotations ({len(ann)}) does not match number of class members"
            f" ({len(class_members)}). Please check that all class members have type annotations and/or a default"
            " value. If you don't want to specify a default value, please use the literal `dataclasses.MISSING`."
        )
    # 遍历注解并为可变类型添加字段工厂
    for key in ann:
        # 在类中查找匹配的字段
        value = class_members.get(key, MISSING)
        # 检查键是否属于 ClassVar
        # 在这种情况下，我们不能使用 default_factory！
        origin = getattr(ann[key], "__origin__", None)
        if origin is ClassVar:
            continue
        # 检查 f 是否为 MISSING
        # 注意：暂时注释掉，因为它会导致 dataclass 继承问题
        #   当父类同时具有位置参数和关键字参数时。
        # 参考：https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
        # TODO: 检查这是否在 Python 3.10 中已修复
        # if f is MISSING:
        #     continue
        if isinstance(value, Field):
            setattr(cls, key, value)
        elif not isinstance(value, type):
            # 为可变类型创建字段工厂
            value = field(default_factory=_return_f(value))
            setattr(cls, key, value)


def _custom_post_init(obj):
    """深拷贝所有元素，以避免 dataclass 初始化时可变对象的共享内存问题。

    该函数被显式调用，而不是作为 :func:`_process_mutable_types()` 的一部分，
    以防止映射代理类型（即映射对象的只读代理）的错误。该错误在使用分层数据类
    进行配置时抛出。
    """
    for key in dir(obj):
        # 跳过双下划线成员
        if key.startswith("__"):
            continue
        # 获取数据成员
        value = getattr(obj, key)
        # 检查注解
        ann = obj.__class__.__dict__.get(key)
        # 复制可变的类成员
        if not callable(value) and not isinstance(ann, property):
            setattr(obj, key, deepcopy(value))


def _combined_function(f1: Callable, f2: Callable) -> Callable:
    """将两个函数组合为一个。

    Args:
        f1: 第一个函数。
        f2: 第二个函数。

    Returns:
        组合后的函数。
    """

    def _combined(*args, **kwargs):
        # 调用两个函数
        f1(*args, **kwargs)
        f2(*args, **kwargs)

    return _combined


"""
辅助函数
"""


def _skippable_class_member(key: str, value: Any, hints: dict | None = None) -> bool:
    """检查在 configclass 处理中是否应跳过该类成员。

    以下成员会被跳过：

    * 双下划线成员：``__name__``、``__module__``、``__qualname__``、``__annotations__``、``__dict__``。
    * 手动添加的特殊类函数：来自 :obj:`_CONFIGCLASS_METHODS`。
    * 已存在于类型注解中的成员。
    * 绑定到类对象或类的函数。
    * 绑定到类对象的属性。

    Args:
        key: 类成员名称。
        value: 类成员值。
        hints: 类的类型提示。默认为 None，此时不检查成员是否存在于类型提示中。

    Returns:
        如果应跳过该类成员则返回 True，否则返回 False。
    """
    # 跳过双下划线成员
    if key.startswith("__"):
        return True
    # 跳过手动添加的特殊类函数
    if key in _CONFIGCLASS_METHODS:
        return True
    # 检查键是否已存在
    if hints is not None and key in hints:
        return True
    # 跳过绑定到类的函数
    if callable(value):
        # FIXME: 这还不能用于静态方法，因为它们本质上被视为函数类型。
        # 检查类方法
        if isinstance(value, types.MethodType):
            return True

        if "CollisionAPI" in value.__name__:
            return False

        # 检查实例方法
        signature = inspect.signature(value)
        if "self" in signature.parameters or "cls" in signature.parameters:
            return True

    # 跳过属性方法
    if isinstance(value, property):
        return True
    # 否则，不跳过
    return False


def _return_f(f: Any) -> Callable[[], Any]:
    """返回用于创建可变/不可变变量的默认工厂函数。

    该函数应用于为变量创建默认工厂函数。

    示例：

        .. code-block:: python

            value = field(default_factory=_return_f(value))
            setattr(cls, key, value)
    """

    def _wrap():
        if isinstance(f, Field):
            if f.default_factory is MISSING:
                return deepcopy(f.default)
            else:
                return f.default_factory
        else:
            return deepcopy(f)

    return _wrap
