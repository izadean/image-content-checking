data_dir=$1

download_and_unzip() {
    local url=$1
    local output_zip=$2
    local dest_dir=$3

    curl -L "$url" > "$output_zip"
    unzip "$output_zip" -d "$dest_dir"
    rm "$output_zip"
}

move_images() {
    local src_dir=$1
    local dest_dir=$2

    mkdir -p "$dest_dir"
    find "$src_dir" -type f -print0 | xargs -0 -I{} mv {} "$dest_dir"
    rm -r "$src_dir"
}

images_dir="$data_dir/images"

download_and_unzip "http://images.cocodataset.org/zips/train2014.zip" "$data_dir/images1.zip" "$data_dir"
download_and_unzip "http://images.cocodataset.org/zips/val2014.zip" "$data_dir/images2.zip" "$data_dir"
download_and_unzip "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "$data_dir/anns.zip" "$data_dir"

move_images "$data_dir/train2014" "$images_dir"
move_images "$data_dir/val2014" "$images_dir"
