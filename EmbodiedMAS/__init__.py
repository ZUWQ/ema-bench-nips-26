# Makes EmbodiedMAS a package so it can be run with `python -m EmbodiedMAS.base_agent`

import importlib.util
import sys
from pathlib import Path

from .env_wrapper import EnvironmentWrapper
from .observation import ObservationCamera
from .observation import PerceptionInfo

# Import from Only_Language-base Agent folder
_ol_agent_path = Path(__file__).parent / "Only_Language-base Agent"
if _ol_agent_path.exists():
    # Add parent directory to sys.path for relative imports to work
    _parent_path = str(_ol_agent_path.parent)
    if _parent_path not in sys.path:
        sys.path.insert(0, _parent_path)
    
    # Import base_agent_only_language
    _base_ol_spec = importlib.util.spec_from_file_location(
        "base_agent_only_language", _ol_agent_path / "base_agent_only_language.py"
    )
    _base_ol_module = importlib.util.module_from_spec(_base_ol_spec)
    _base_ol_spec.loader.exec_module(_base_ol_module)
    BaseAgentOnlyLanguage = _base_ol_module.BaseAgentOnlyLanguage
    
    # Import action_ol
    _action_ol_spec = importlib.util.spec_from_file_location(
        "action_ol", _ol_agent_path / "action_ol.py"
    )
    _action_ol_module = importlib.util.module_from_spec(_action_ol_spec)
    _action_ol_spec.loader.exec_module(_action_ol_module)
    ActionAPI_ol = _action_ol_module.ActionAPI
    
    # Import llm_agent_ol from Decentralized MAS subfolder
    _dmas_path = _ol_agent_path / "Decentralized MAS" / "llm_agent_ol.py"
    if _dmas_path.exists():
        _llm_ol_spec = importlib.util.spec_from_file_location(
            "llm_agent_ol", _dmas_path
        )
        _llm_ol_module = importlib.util.module_from_spec(_llm_ol_spec)
        _llm_ol_spec.loader.exec_module(_llm_ol_module)
        LLMAgent_ol = _llm_ol_module.LLMAgent
        LLMAgentFire_ol = _llm_ol_module.LLMAgentFire
        LLMAgentSave_ol = _llm_ol_module.LLMAgentSave
        LLMController = _llm_ol_module.LLMController
    else:
        LLMAgent_ol = None
        LLMAgentFire_ol = None
        LLMAgentSave_ol = None
        LLMController = None
else:
    ActionAPI_ol = None
    BaseAgentOnlyLanguage = None
    LLMAgent_ol = None
    LLMAgentFire_ol = None
    LLMAgentSave_ol = None
    LLMController = None

# Import from Vision_Language-base_Agent (VL single-agent module)
_v_agent_path = Path(__file__).parent / "Vision_Language-base_Agent" / "Single_agent"
if _v_agent_path.exists():
    # Import base_agent_vision_language (parent of Single_agent)
    _vl_root = _v_agent_path.parent
    _base_v_spec = importlib.util.spec_from_file_location(
        "base_agent_vision_language", _vl_root / "base_agent_vision_language.py"
    )
    _base_v_module = importlib.util.module_from_spec(_base_v_spec)
    _base_v_spec.loader.exec_module(_base_v_module)
    BaseAgentVision = _base_v_module.BaseAgentVisionLanguage
    
    # Import action_vl
    _action_v_spec = importlib.util.spec_from_file_location(
        "action_vl", _vl_root / "action_vl.py"
    )
    _action_v_module = importlib.util.module_from_spec(_action_v_spec)
    _action_v_spec.loader.exec_module(_action_v_module)
    ActionAPI_v = _action_v_module.ActionAPI
    
    # Import llm_agent_vl
    _llm_v_spec = importlib.util.spec_from_file_location(
        "llm_agent_vl", _v_agent_path / "llm_agent_vl.py"
    )
    _llm_v_module = importlib.util.module_from_spec(_llm_v_spec)
    _llm_v_spec.loader.exec_module(_llm_v_module)
    LLMAgent_v = _llm_v_module.LLMAgent
    LLMAgentFire_v = _llm_v_module.LLMAgentVisionLanguageFire
    LLMAgentSave_v = _llm_v_module.LLMAgentVisionLanguageSave
else:
    ActionAPI_v = None
    BaseAgentVision = None
    LLMAgent_v = None
    LLMAgentFire_v = None
    LLMAgentSave_v = None

# For backward compatibility, use the vision versions as default if available
# Otherwise use the only-language versions
LLMAgent = LLMAgent_v if LLMAgent_v is not None else LLMAgent_ol
LLMAgentFire = LLMAgentFire_v if LLMAgentFire_v is not None else LLMAgentFire_ol
LLMAgentSave = LLMAgentSave_v if LLMAgentSave_v is not None else LLMAgentSave_ol
ActionAPI = ActionAPI_v if ActionAPI_v is not None else ActionAPI_ol

__all__ = [
	"EnvironmentWrapper",
	"BaseAgentOnlyLanguage",
	"LLMAgent",
	"LLMAgentFire",
	"LLMAgentSave",
	"BaseAgentVision",
	"ObservationCamera",
	"SimpleMapBuilder",
	"PerceptionInfo",
	"ActionAPI",
]