from crewai import Task
from textwrap import dedent



class InformationCollectionTasks():
  pass

class Impax10StepWritingTasks():
  pass

class ESGAnalysisTasks():
  pass

class AuxiliaryTasks():
  def get_ticker(self, agent, company: str):
    task = Task(
      description=dedent("""Your task is to get the company's ticker based on 
       the name user input.""",
      expected_output="Ticker that can be queried. For example, 'Apple Inc.' should return 'AAPL'.")
      )

class StockAnalysisTasks():
  # def write_10_step(self, agent, company) -> List[Task]:
  def research(self, agent, company):
    return Task(description=dedent(f"""
        Collect and summarize company filings, third party data, research 
        reports, recent news articles, press releases, and market analyses 
        related to the stock and its industry you can find.
        Try to use the latest information available, especially for filings.
        Try to use the `Search SEC Filings` tool for filings. You can
        change your query terms to get the results you want, such as 
        `AAPL|company history`, `AAPL|main business`. DO NOT use queries like
        `AAPL|latest 10-K` as this will return nothing useful. 
                                   
        Following the following template:
        <template> The “10-step” Analysis Template							<Company Name>

          1.	Company Snapshot & Investment Thesis
          Give a snapshot of the company's history, main businesses, and segments.
          What are the evidences that establish the company's role in the transition to a more sustainable economy? 
          You can consider its key businesses and their impact on the environment and the society. 
          Why is an investment in the company an attractive opportunity?

          2.	Market
          How is the market that the company operates in defined with respect to size, regulation, level of competition, barriers to entry, and growth? 
          Conduct a porter's five forces analysis.
          Describe the competitive landscape and the company’s position in the addressable market, together with customers and customer concentration, and suppliers? 

          3.	Competitive Advantage
          What unique technologies, brand strength, embedded intellectual property, scale and distribution capabilities exist that give the business a competitive edge?

          4.	Business Model and Strategy Analysis
          Does the company have a sustainable competitive advantage? Are the company’s plans credible? Are the financial returns satisfactory or is there a plan to improve these?

          5.	Risks
          What are the perceived risks of investing in the context of the wider landscape (industry dynamics, policy, global macro factors and societal forces), from the perspective of different stakeholders and from the perspective of the company’s supply chain and distribution capability?

          6.	ESG 
          Does the board follow recognized best practices and is there enough independence? Are there areas of concern regarding corporate governance, controversies at the company, financial disclosure, health and safety, environmental or social considerations? Does the company comply with UN Global Compact principles? Has a WorldCheck sanctions report been run?

          7.	Management
          How much experience does the current management team have and how effective have they been? Are there succession risks?

          8.	Valuation (ask your coworker Financial Analyst)
          Financial statement analysis leading to a medium-term fair value assessment of the company. Are the shares trading at a discount? How does the value compare to history and peers?

          9.	Trading (ask your coworker Financial Analyst)
          Which security has the liquidity, if more than one? Is there sufficient liquidity to establish an appropriate allocation within the portfolio? 

          10.	Catalysts
          What is the route map for a return on investment?
        Your final answer MUST be a report that includes a
        comprehensive summary of the areas covered in the template.
        Also make sure to return the stock ticker.
        
        {self.__tip_section()}
  
        Make sure to use the most recent data as possible.
  
        Selected company: {company}
      """),
      expected_output=dedent("Detailed, accurate, well throughtout research "
              "covering all points mentioned above "
              "Quote your sources whenever possible. "),
      agent=agent,
      output_file=f'outputs/10-step-reports/{company}_qualitative_analysis.md'
    )
  
  def esg_analysis(self, agent, company):
    task = Task(description=dedent(
      f"""Conduct a thorough ESG analysis of {company}. The most relevant document
      typically is the ESG report. Try to search the link of the document.
        Consider the following:
      
        Corporate Governance - Break down into the following categories:
        - Board structure (board/committee independence, tenure profile, shareholder rights, and internal control)
        - Compensation: does the company have good disclosure on this? What are the KPIs and is ESG integrated?
        - Shareholder rights: What protection mechanisms are available on shareholder rights?
        - Internal controls: what internal controls are there?
        - Governance of sustainability: What governance bodies are set up for sustainability?

        Diversity, inclusion, human capital management:
        - Diversity in leadership: how many percentage of the board/exectutive team are women/minorities? Do they have targets?
        - Workplace equality: What measures are in place to make sure of workplace equality?
        - Human capital management: What hiring/retention/employee engagement practices are there?

        Climate change:
        - Does the company report its climate transition risks (scope 1-3 carbon emission and targets of this?)? Does it report to CDP?
        - What physical climate risk analysis has it done? Does it follow TCFD recommendations?
        
        Material environmental & social risks
        - What materialities has it identified? How is it managing these risks?

        Controversies & incidents - any controversies at all?
    """),
    agent=agent,
    expected_output=dedent("""A well written report covering all aspects of ESG 
                           as specified."""),
    output_file=f'outputs/10-step-reports/{company}_esg_analysis.md'
    # TODO - add worldcheck and UNGC tools
    )
    return task

  def financial_analysis(self, agent, company): 
    return Task(description=dedent(f"""
        Conduct a thorough analysis of {company}'s financials. Focus on the 
        quantitative aspects. Your teammate will take care of the qualitatives.
        Try to come up with a fair value assessment of the company's valuation,
        using either DCF, DDM, or multiple-based approaches.
        {self.__tip_section()}

        Make sure to use the most recent data possible.
        
      """),
      expected_output=dedent("""
        Your final report MUST expand on the summary provided
        but now including a clear assessment of the stock's
        financial standing, its strengths and weaknesses, 
        and how it fares against its competitors in the current
        market scenario."""),
      agent=agent,
      output_file=f'outputs/10-step-reports/{company}_financial_analysis.md'
    )

  def filings_analysis(self, agent, company):
    return Task(description=dedent(f"""
        Analyze the latest filings from EDGAR for
        the stock in question: {company}. 
        Try to use the `Search SEC Filings` tool. You can consecutively query
        by changing your query terms, such as doing `AAPL|What is the management
        discussion`, followed by `AAPL|What are the product segments`. 
        Focus on key sections like Management's Discussion and
        Analysis, financial statements, management teams, 
        and any disclosed risks. Do not be too generic in your queries.
        Extract relevant data and insights that could influence
        the stock's future performance.
        Try to be as detailed as possible.
        {self.__tip_section()}        
      """),
      expected_output=dedent("""Your final answer must be an expanded report that now
        also highlights significant findings from these filings,
        including any red flags or positive indicators for
        your customer."""),
      agent=agent,
      output_file=f'outputs/10-step-reports/{company}_filining_analysis.md'
    )

  def edit(self, agent, company):
    return Task(description=dedent(f"""
        Review and synthesize the analyses provided by the
        Financial Analyst and the Research Analyst.
        Combine these insights to form a comprehensive 
        investment recommendation. Format the report so it's easy to read.
        Try to keep the details provided by your coworkers, while still being
        succinct.
        You MUST Consider all aspects, both qualitative and quantitative.
        {self.__tip_section()}
      """),
      expected_output=dedent(f"""A well-written and formatted report following the 
      following template about {company}:
      <template> The “10-step” Analysis Template							<Company Name>
      1.	Company Snapshot & Investment Thesis
      2.	Market
      3.	Competitive Advantage
      4.	Business Model and Strategy Analysis
      5.	Risks
      6.	ESG 
      7.	Management
      8.	Valuation
      9.	Trading 
      10.	Catalysts
      </template>
      """),
      agent=agent,
      output_file=f'outputs/10-step-reports/{company}_final.md'
    )

  def __tip_section(self):
    return "BEST WORK is always encouraged. You are up for a promotion, "
  "and good work here can really help your cause. "
