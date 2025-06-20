from langgraph.graph import StateGraph, START

from src.constants import RETRIEVE, QNA_CHATBOT, UPDATE_PROFILE, UPDATE_INSTRUCTIONS
from src.nodes.retrieve import retrieve
from src.nodes.qna_chatbot import metta_chatbot
from src.nodes.update_profile import update_profile
from src.nodes.update_instruction import update_instructions
from src.nodes.route_message import route_message
from src.states import QnAChatbotState


builder = StateGraph(QnAChatbotState)
builder.add_node(RETRIEVE, retrieve)
builder.add_node(QNA_CHATBOT, metta_chatbot)
builder.add_node(UPDATE_PROFILE, update_profile)
builder.add_node(UPDATE_INSTRUCTIONS, update_instructions)

builder.add_edge(START, RETRIEVE)
builder.add_edge(RETRIEVE, QNA_CHATBOT)
builder.add_conditional_edges(QNA_CHATBOT, route_message)
builder.add_edge(UPDATE_INSTRUCTIONS, QNA_CHATBOT)

app = builder.compile()

# Skip draw graph
# app.get_graph().draw_mermaid_png(output_file_path="graph.png")