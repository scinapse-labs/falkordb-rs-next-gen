import os
import sys
from functools import wraps

from RLTest import Env as Environment, Defaults

import redis
from redis import ResponseError
from falkordb import FalkorDB, Graph, Node, Edge, Path, ExecutionPlan

from base import FlowTestsBase

Defaults.decode_responses = True

SANITIZER     = os.getenv('SANITIZER', '')      != ''
CODE_COVERAGE = os.getenv('CODE_COVERAGE', '0') == '1'

def Env(moduleArgs=None, env='oss', useSlaves=False, enableDebugCommand=False, shardsCount=None):
    env = Environment(decodeResponses=True, moduleArgs=moduleArgs, env=env,
                      useSlaves=useSlaves, enableDebugCommand=enableDebugCommand, shardsCount=shardsCount)
    db  = FalkorDB("localhost", env.port)
    return (env, db)

def skip():
    def decorate(f):
        @wraps(f)
        def wrapper(x, *args, **kwargs):
            env = x if isinstance(x, Environment) else x.env
            env.skip()
            return f(x, *args, **kwargs)
        return wrapper
    return decorate
