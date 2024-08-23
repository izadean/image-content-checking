MODEL_STORE="deployment/model-store"
MODELS=""

for file in "$MODEL_STORE"/*.mar; do
    filename=$(basename "$file" .mar)
    MODELS+="$filename=$file "
done

MODELS="${MODELS% }"

torchserve --start \
    --ncs \
    --ts-config deployment/config.properties \
    --model-store "$MODEL_STORE" \
    --models "$MODELS"