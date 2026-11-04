\# DiaCare – GitHub Copilot Instructions



\## 1. Project Overview



DiaCare is a Streamlit-based multi-agent web application for diabetes risk assessment and medical knowledge support.



Main technologies:

\- Python 3.10

\- Streamlit

\- PyTorch MLP

\- Neo4j AuraDB

\- Sentence Transformers and RAG

\- Gemini API

\- Q-Learning

\- Rule Engine

\- Pytest



\## 2. Mandatory Development Rules



\- Always inspect existing code before modifying it.

\- Reuse existing modules and functions whenever possible.

\- Do not rewrite the project from scratch.

\- Do not replace Streamlit with another framework.

\- Do not delete working functionality.

\- Do not create duplicate files or modules.

\- Do not change existing APIs or data structures without necessity.

\- Do not add unnecessary dependencies.

\- Modify only files required for the task.

\- Keep code simple, readable and maintainable.



\## 3. Architecture Rules



Preserve the existing Multi-Agent architecture:

\- Coordinator Agent

\- Prediction Agent

\- Knowledge Agent

\- Support Agent



Responsibilities:

\- Prediction Agent: MLP-based risk assessment.

\- Knowledge Agent: Neo4j-RAG medical knowledge retrieval.

\- Support Agent: user-facing response generation.

\- Coordinator Agent: workflow coordination.



Q-Learning is used only for Agent routing.



Neo4j stores medical and food knowledge.

SQLite may store personal records such as glucose logs and medication schedules.



Gemini API must use retrieved evidence when answering medical questions.



Do not add new Agents unless technically necessary.



\## 4. Project Scope



In-scope features:

1\. Diabetes risk assessment.

2\. Medical knowledge retrieval using Neo4j-RAG and Gemini.

3\. Smart glucose diary.

4\. Medication reminders.

5\. AI Meal Planner and Glycemic Index Tracker.

6\. Multi-Agent orchestration using Q-Learning.



Out of scope:

\- Healthcare-worker role or dashboard.

\- Q-Learning exercise recommendations.

\- Glucose meter or CGM hardware integration.

\- Automatic medication prescriptions or dose adjustments.



\## 5. Medical Safety Rules



\- Never claim to diagnose diabetes.

\- Clearly distinguish model output from clinical diagnosis.

\- Never present an uncalibrated model score as a clinically validated probability.

\- Never prescribe or adjust medication or insulin doses.

\- Never infer medication doses from glucose readings.

\- Use credible medical sources.

\- Never fabricate medical references or citations.

\- Clearly disclose insufficient evidence.

\- Treat nutrition and GI values as source-dependent information.



\## 6. Security Rules



\- Never hardcode API keys or passwords.

\- Read secrets from environment variables.

\- Keep `.env` out of Git.

\- Update `.env.example` using variable names only, without real credentials.

\- Never expose secrets in logs or the user interface.



\## 7. Implementation and Testing



\- Implement changes incrementally.

\- Check dependent modules before modifying shared interfaces.

\- Validate user input.

\- Handle errors and unavailable services.

\- Run relevant tests after each change.

\- Run `pytest -v` when appropriate.

\- Do not claim a feature is complete without implementation and testing.

\- Update documentation only when actual implementation status changes.



\## 8. Working Style



\- Prefer direct implementation over lengthy planning.

\- Avoid unnecessary analysis and documentation.

\- Do not repeat completed work.

\- Report changed files and test results concisely.

\- Ask before destructive changes or major architectural modifications.



Always prioritize a working, safe and maintainable DiaCare application.

