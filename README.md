# Bedrock-multimodal
Repo for adding multiple LLM's for bedrock use instead of one. Also using streamlit for front end for easier front end. Temporarily using so work can be done and I can share what I can in order to collaborate with some teammates. This is a rushed project so far so please bear with me on gathering information, explaining the layout, and fixing any issues you find. 

# How to use:
Clone repo including requirement.txt file, and run "streamlit run front-end-2.py" to run streamlit instance locally with required dependencies. This is how you can make changes to front end. For backend changes we will have to work something out once we get access to a proper backend.

# Issues/Known Bugs:
- There is a known bug for one of the models ("Meta LLaMA3 2-1B Instruct"). The issue is linked to the way the json payload is being sent to the model. Need to change the "format" of the payload to include expected syntactical format said model.
- Currently fixing "timestamps" column in the database that the API calls from AWS lambda. (This currently why you'll likely get dates/table responses that differ from your "start" and "end" times at the moment.
- Fixing prompt box to also allow file uploads. (In my testing in a staging environment, file uploads can be up to 200MB in size, but also may be limited by the server/web engine you deploy the backend on (i.e. Nginx, Apache, etc. have additional options to configure in order to allow maximum file sizes to be uploaded).
- Working on documenting API information for sending and recieving RESTFUL responses. This is specific to this local repo, but being noted in case we decide to build off the current design in ay way until we have access to any internal API's that may be availible. FastAPI is another option that has the benefit of building out API documentation "automagically" as you build out your resources etc., but has other considerations/additinal overhead.

# Potential Features/Topics of Discussion: 
Explore only using Converse API:
AWS Bedrock has an API for talking/"conversing" with models. Currently, The code is using multiple "formats" to talk to each different model separtely that requires code that formats json payloads to meet each model's expected syntactical format. Using AWS converse API may be able to:

1. Help eliminate the need to add additional code/spikes to adjust to any additional models that we want to test.
2. Utilize an API meant for persistent conversational sessions with a model.
3. Make it easier to save sessions locally (or to a sharedrive, S3, etc) and potentially be able to share previous sessions up for models to load into memory etc.
 

Architectural System Design & Framework Discussion:
- Need to follow up with teammates to get their feedback and opinions on what they want to use are most comfortable with. None of what I have done in this project has to be used and having collaborative feedback on how we would like to proceed on top of what we've done so far within sprints will be nice for adjustments moving forward.

Adding RAG options:
- Working on deploying an AWS Bedrock Agent that will be able to determine "when" to hit my "rf_measurements" API based on user's prompts.
- There are other options to deploy "RAG" as well like Redshift, utilizing knowledge bases, etc., but those are pretty expensive and not ideal in my opinion to meet requirements in the way we discussed with product owner. 
