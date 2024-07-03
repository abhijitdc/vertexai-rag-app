from google.cloud import discoveryengine_v1alpha as discoveryengine
from datastorehelper import SearchHelper


class GroundingChecker:
    def __init__(self):
        self.project_id = "[PROJECT_ID]"
        self.client = discoveryengine.GroundedGenerationServiceClient()

    def check_grounding(self, answer_candidate_text, fact_docs):

        local_grounding_config = self.client.grounding_config_path(
            project=self.project_id,
            location="global",
            grounding_config="default_grounding_config",
        )

        input_facts = []
        for fact in fact_docs:
            input_facts.append(
                discoveryengine.GroundingFact(
                    fact_text=str(fact.page_content),
                )
            )

        request = discoveryengine.CheckGroundingRequest(
            grounding_config=local_grounding_config,
            answer_candidate=answer_candidate_text,
            facts=input_facts,
            grounding_spec=discoveryengine.CheckGroundingSpec(citation_threshold=0.6),
        )

        response = self.client.check_grounding(request=request)
        return response


if __name__ == "__main__":
    dsHelper = SearchHelper()
    facts_docs = dsHelper.searchWithRanksDocs(
        "What is vaping and how it affects the teens?"
    )
    answer_candidate_text = """Vaping can cause nicotine dependence and harm a young person's brain, which is still developing until their mid to late 20s. Vaping can harm parts of the brain that control attention, learning and memory. It can also increase the chances of using other addictive substances. The nicotine in vapes is highly addictive and may also affect their mental health."""

    grounding_checker = GroundingChecker()
    results = grounding_checker.check_grounding(answer_candidate_text, facts_docs)
    print(results)
