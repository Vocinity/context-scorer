import dataclasses
import enum

import context_scorer_pb2 as cs
import context_scorer_pb2_grpc as csrpc
import grpc
import os
from typing import List

#python3 -m grpc_tools.protoc --proto_path=/opt/projects/context-scorer/grpc-server/src/ --python_out=. --grpc_python_out=. /opt/projects/context-scorer/grpc-server/src/context-scorer.proto

@dataclasses.dataclass
class Homonym_Generation_Query:
    input: str
    max_num_of_best_homophonic_alternatives: int
    max_distance: int
    dismissed_word_indices: List[int]
    dismissed_words: List[str]


@dataclasses.dataclass
class Context_Scoring_Query:
    pre_context: str
    input: Homonym_Generation_Query
    post_context: str
    per_char_normalized: bool
    model_code: str = ""

class Context_Scorer_Client(object):
    def __init__(self, host="127.0.0.1", port=1991):
        self.host = host
        self.port = str(port)
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.port))
        self.stub = csrpc.Context_Scorer_gRPCStub(self.channel)

    def say_hi(self, models:List[str]=[]):
        models_to_subscribe = cs.Knock_Knock()
        # if you dont state model code, generic one (codename is generic, only one we have right now) will be used
        if not len(models):
            models_to_subscribe.models_that_you_planning_to_use.add()
        for model_code in models:
            model = models_to_subscribe.models_that_you_planning_to_use.add()
            model.code = model_code
        self.stub.say_hi(models_to_subscribe)

    def get_homonyms(self, query: Homonym_Generation_Query):
        request = cs.Homonym_Generation_Query()
        request.input = query.input
        request.max_num_of_best_homophonic_alternatives = query.max_num_of_best_homophonic_alternatives
        request.max_distance = query.max_distance
        request.dismissed_word_indices.extend(query.dismissed_word_indices)
        request.dismissed_words.extend(query.dismissed_words)
        request.model_code = query.model_code
        response = self.stub.get_homonyms(request)
        for alternative in response.alternatives_of_input:
            print(alternative)

    def get_best_n_alternatives(self, query: Context_Scoring_Query):
        request = cs.Context_Scoring_Query()
        request.material.pre_context = query.pre_context
        request.material.input.input = query.input.input
        request.material.input.max_num_of_best_homophonic_alternatives = query.input.max_num_of_best_homophonic_alternatives
        request.material.input.max_distance = query.input.max_distance
        request.material.input.dismissed_word_indices.extend(query.input.dismissed_word_indices)
        request.material.input.dismissed_words.extend(query.input.dismissed_words)
        request.material.per_char_normalized = query.per_char_normalized
        response = self.stub.get_best_n_alternatives(request)
        for score in response.scores:
            print(score)


if __name__ == '__main__':
    client = Context_Scorer_Client("10.0.0.39",8081)
    try:
        client.say_hi()
    except grpc.RpcError as e:
        print(e.code())
    query = Context_Scoring_Query(
        pre_context="Good Morning. My name is Valerie, and I can answer your questions about the Echelon home gym products. You may interrupt me by saying Hey Valerie. And whenever you say something, please give me a moment to gather my thoughts. Also, opening the icon tray below will pause our conversation, and closing it will continue it. Calls may be recorded for training and quality assurance purposes.  Before we get started, here are some helpful phrases you can use during our chat. Click on the eye in the icon tray to pick your product of interest or please say connected bike, smart rower, or reflect mirror.",
        input=Homonym_Generation_Query(input="smart rower.", max_num_of_best_homophonic_alternatives=5, max_distance=2,
                                       dismissed_word_indices=[],
                                       dismissed_words=[]), post_context="", per_char_normalized=True)
    client.get_best_n_alternatives(query=query)
