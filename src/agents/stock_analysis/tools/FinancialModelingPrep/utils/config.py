from __future__ import annotations
from types import SimpleNamespace
from typing import Tuple, Any, Dict, List, Optional, Union

class Config(SimpleNamespace):
    def __init__(self, d: Optional[Dict]=None, **kwargs):
        if not d: d = dict()
        super(Config, self).__init__(**d, **kwargs)

    def get(self, name, returntype: Union[None, str]='dict', 
        default: Any=None, ):
        """
        :param name: The name of the attribute to get.
        :param returntype: The typpe of the returned attribute. Will force 
            returned object into the specified type.
            Takes strings 'dict' and 'bool'
        :param default: The default value to return if the attribute is not found
        """
        if hasattr(self, name):
            res = getattr(self, name)
            if returntype == 'bool':
                return True if res else False
            elif returntype == 'dict':
                return vars(res)
            elif returntype == 'list':
                return list(res)
            elif returntype == 'str':
                return str(res)
            elif returntype == 'int':
                return int(res)
            elif returntype == 'float':
                return float(res)
            else:
                raise NotImplementedError("returntype must be 'dict' or 'bool'")
        else:
            return default