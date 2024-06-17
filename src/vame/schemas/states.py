from functools import wraps
from pydantic import BaseModel, Field
from typing import Optional, Dict
from pathlib import Path
import json
from enum import Enum

class StatesEnum(str, Enum):
    success = 'success'
    failed = 'failed'


class BaseStateSchema(BaseModel):
    config: str = Field(title='Configuration file path')
    execution_state: StatesEnum | None = Field(title='Method execution state', default=None)



class EgocentricAlignmentFunctionSchema(BaseStateSchema):
    pose_ref_index: list = Field(title='Pose reference index', default=[0, 5])
    crop_size: tuple = Field(title='Crop size', default=(300, 300))
    use_video: bool = Field(title='Use video', default=False)
    video_format: str = Field(title='Video format', default='.mp4')
    check_video: bool = Field(title='Check video', default=False)



class VAMEPipelineStatesSchema(BaseModel):
    egocentric_alignment: Optional[EgocentricAlignmentFunctionSchema | Dict] = Field(title='Egocentric alignment', default={})
    csv_to_numpy: Optional[dict] = Field(title='CSV to numpy', default={})


def _save_state(model: BaseModel, function_name: str, state: StatesEnum) -> None:
    """
    Save the state of the function to the project states json file.
    """
    config_file_path = Path(model.config)
    project_path = config_file_path.parent
    states_file_path = project_path / 'states/states.json'

    with open(states_file_path, 'r') as f:
        states = json.load(f)

    pipeline_states = VAMEPipelineStatesSchema(**states)
    model.execution_state = state
    setattr(pipeline_states, function_name, model.model_dump())

    with open(states_file_path, 'w') as f:
        json.dump(pipeline_states.model_dump(), f, indent=4)


def save_state(model: BaseModel):
    """
    Decorator responsible for validating function arguments using pydantic and
    saving the state of the called function to the project states json file.
    """
    def decorator(func: callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create an instance of the Pydantic model using provided args and kwargs
            function_name =  func.__name__
            attribute_names = list(model.model_fields.keys())

            kwargs_dict = {}
            for attr in attribute_names:
                if attr == 'execution_state':
                    kwargs_dict[attr] = 'running'
                    continue
                kwargs_dict[attr] = kwargs.get(attr, model.model_fields[attr].default)

            # Override with positional arguments
            for i, arg in enumerate(args):
                kwargs_dict[attribute_names[i]] = arg
            # Validate function args and kwargs using the Pydantic model.
            kwargs_model = model(**kwargs_dict)
            try:
                func_output = func(*args, **kwargs)
                _save_state(kwargs_model, function_name, state=StatesEnum.success)
                return func_output
            except Exception as e:
                _save_state(kwargs_model, function_name, state=StatesEnum.failed)
                return func_output
        return wrapper
    return decorator