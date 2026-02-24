# Decision Log — Local RAG Pipeline

## Model Selection
- **Chosen LLM:** qwen3:8b
- **Why:** Overall qwen3:8b scored higher accross different questions, it has 8b parameters so I expected it, compared to phi4-mini which has ~1.3. Precision very important for RAGs.
- **What I considered but rejected:** Tested phi4-mini, it was way quicker, but outputs were not always right and sometimes lacked reasoning. phi4-mini does not meet the reliability requirements for this task.

## Embedding Model
- **Chosen:** nomic-embed-text
- **Why:** Looked here(https://ollama.com/search?c=embedding) and saw that it had most downloads so it means people are using it. Also it was suggested in the task description.

## Chunking Strategy
- **Chunk size:** 512
- **Overlap:** 50
- **Why:** Experimented with 256, 512 and 1024. 256 was too small, it split related content mid sentence. 1024 threw some off topic answers. 512 offered the best balance, it kept context in one chunk and answers were precise.
  
## Retrieval Configuration
- **Top-K:** 10
- **Why:** With 5 some questions that required synthesizing information from multiple chunks were failing because relevant chunks were not in the top 5. I increased to 10 which provided a larger candidate pool where the correct answers could be found.

## Observations
- **What worked well:** Examples below
- **What failed:** Examples below
- **Local vs cloud expectations:** Azure OpenAI would definitely work quicker and provide better multi step reasoning. However since the data is company only sending it overseas to cloud APIs is not acceptable.

## If I Had More Time / Better Hardware
- I would use a bigger model for better reasoning and experimented with more pdf readers, since pypdf cannot read pdf tables. Also would have tested much more questions and tested how bulletproof the RAG is. Would have tried to make model hallucinate.



### What worked well
  **1.** Query: What is the salary range for software engineers at NovaTech?

Answer:
I cannot find this information in the provided documents. The context only mentions that the average engineering salary in Lisbon is 35-40% lower than in Munich, but it does not specify absolute salary ranges for software engineers at NovaTech.

Comment: Very good, unanswerable question without hallucination.

**2.** Query: If a security incident is discovered involving unauthorized access to customer payment data, what are the required notification timelines and who must be notified?

Answer:
If a security incident involves unauthorized access to customer payment data (a P1 incident), the required notification timelines and recipients are as follows:

(4 bullet points with all needed info)

**Key Parties Notified**: Customers, supervisory authorities (per GDPR), and internal stakeholders (CISO, Security Operations team).

**3.** 
Query: A new employee is struggling with a sensitive workplace issue they don't want to raise with their manager. What formal and informal channels does NovaTech offer them?

Answer:
NovaTech offers both formal and informal channels for addressing sensitive workplace issues:

### **Informal Channels**
- **Discuss with a buddy**: Employees can raise concerns informally with their manager or a buddy (colleague). Buddies are expected to maintain confidentiality within company policy boundaries.

### **Formal Channels**
- **Submit a formal grievance**: If informal resolution isn’t possible, employees can escalate the issue formally via: 

  - The **NovaHR portal**
  - The **NovaTrust Reporting Line** (anonymous reporting option)

These channels ensure confidentiality and support for employees hesitant to discuss issues directly with their manager.

Combined info from different sections of the handbook

**4.** Query: What is the NovaTech Copilot initiative and what is the total approved budget?

Answer:
The NovaTech Copilot initiative is an AI-powered conversational assistant designed to operate across ProjectHub and Insight, enabling users to query project status, generate reports, create tasks, and receive proactive recommendations through natural language interaction. The total approved budget for the initiative is **EUR 1.2 million**.

### What failed
**1.** Query: What are the uptime SLAs for each pricing tier?

Answer:
I cannot find this information in the provided documents. The context mentions "Service Level Agreements" and "uptime guarantees based on tier" in Source [5], but the specific uptime SLA percentages or downtime metrics for each pricing tier are not fully provided in the text.

**2.** Query: How does the Shared Analytics Dashboard differ between NovaTech ProjectHub and NovaTech Insight?

Answer:
I cannot find this information in the provided documents. The context provided does not mention NovaTech ProjectHub or compare its Shared Analytics Dashboard to NovaTech Insight.

256 and 512 chunk size failed

**3.** 
Query: What happens to data if a customer downgrades from Professional to Starter?

Answer:
I cannot find this information in the provided documents.

256 and 512 chunk size failed