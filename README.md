# Image Content Checking

## **Project Definition**
ML project for training binary classification model, takes an image and description and predicts the similarity between them

## **Start Training**: 
1. In the terminal, run the following command: `make train batch_size=<batch_size> epochs=<epochs> learning_rate=<learning_rate>`

## **Start Inference Server**: 
1. Get the run IDs of the models that you want to serve
2. For each model run in your terminal: `make dist run_id=<your_model_run_id> model_name=<the_name_of_model_endpoint>`
3. Run in your terminal: `make serve`
