model_name=$1
serialized_file=$2

HANDLER_FILE=deployment/handler.py
MODEL_FILE=src/model.py
MODEL_STORE=deployment/model-store

mkdir -p $MODEL_STORE
sudo apt install --no-install-recommends -y openjdk-11-jre-headless
pip install torchserve torch-model-archiver
torch-model-archiver --model-name "$model_name" \
                     --version 1.0 \
                     --model-file $MODEL_FILE \
                     --serialized-file "$serialized_file" \
                     --handler $HANDLER_FILE
mv "$model_name" $MODEL_STORE
