from langgraph.graph import START, END, StateGraph
from src.graph import GraphState, Node, Edge


class RagGraph:
    def __init__(self):
        self.node = Node()
        self.edge = Edge()

    def build(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("websearch", self.node.web_search)
        workflow.add_node("retrieve", self.node.retrieve)
        workflow.add_node("grade_document", self.node.grade_documents)
        workflow.add_node("generate", self.node.generate)

        workflow.add_conditional_edges(START, self.edge.route_question, {
            "vectorstore": "retrieve",
            "websearch": "websearch"
        })
        workflow.add_edge("retrieve", "grade_document")
        workflow.add_edge("websearch", "generate")
        workflow.add_conditional_edges("grade_document", self.edge.decide_to_generate, {
            "websearch": "websearch",
            "generate": "generate"
        })
        workflow.add_conditional_edges("generate", self.edge.verify_generations, {
            "relevant": END,
            "irrelevant": "websearch",
            "not supported": "generate"
        })

        return workflow.compile()


if __name__ == "__main__":
    app = RagGraph().build()
    question = "What is agent memory?"
    result = app.invoke({"question": question})
    print(result["generation"])
