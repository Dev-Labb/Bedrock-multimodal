# Bedrock-multimodal
Repo for building a prototype for Bedrock access and API integration.

# How to use:
Clone repo including requirement.txt file, and run "streamlit run front-end-2.py" (acting essentially my "main.py" file for now) to run streamlit instance locally with required dependencies. This is how you can make changes to front end. For backend changes we will have to work something out once we get access to a proper backend.

#Addtional Context:

- The lambda folder contains the AWS lambda function code that interfaces on behalf of users to my backend postgres db. The "current-working-lambda.py" is the one I currently have in place that "works" (with some issues noted below) and the "newer-dynamic-query-lambda.py" is the one I want to implement as it is better scrutured for agentic tooling on top of having dynamic querying capabilities.
- The logged_data-v3.py is just generic dummmy data from kaggle. I actually want to scrap it completely and user "faker" library to generarte a better table to more closely simulate data in BA.
- The openapi.yaml is a slighltly incomplete yaml file that depicts how my api will work once I get the aforementioned newer lambda file working correctly.

# Issues/Known Bugs:
- There is a known bug for one of the models ("Meta LLaMA3 2-1B Instruct"). The issue is linked to the way the json payload is being sent to the model. Need to change the "format" of the payload to include expected syntactical format said model.
- Currently fixing "timestamps" column in the database that the API calls from AWS lambda. (This currently why you'll likely get dates/data inside the dataframe table responses that differ from your "start" and "end" times at the moment).
- Fixing prompt box to also allow file uploads. (In my testing in a staging environment, file uploads can be up to 200MB in size, but also may be limited by the server/web engine you deploy the backend on (i.e. Nginx, Apache, etc. have additional options to configure in order to allow maximum file sizes to be uploaded).
- Working on documenting API information for sending and recieving RESTFUL responses. This is specific to this local repo, but being noted in case we decide to build off the current design in ay way until we have access to any internal API's that may be availible. FastAPI is another option that has the benefit of building out API documentation "automagically" as you build out your resources etc., but has other considerations/additinal overhead.

# Potential Future Features/Topics of Discussion: 
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
