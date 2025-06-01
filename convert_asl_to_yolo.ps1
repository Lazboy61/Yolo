# === Configuratie ===
$source_dir = "D:\Users\Gebruiker\Documents\Project D\Yolo\archive\asl_alphabet_train\asl_alphabet_train"
$output_dir = "D:\Users\Gebruiker\Documents\Project D\Yolo\asl_yolo_small"
$letters = "I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"

# Maak outputfolders aan
$folders = @("images\train", "images\val", "labels\train", "labels\val")
foreach ($folder in $folders) {
    $path = Join-Path $output_dir $folder
    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Force -Path $path
    }
}

# Start met class ID = 0
$class_id = 0

foreach ($letter in $letters) {
    $class_path = Join-Path $source_dir $letter
    $images = Get-ChildItem -Path $class_path -Filter "*.jpg"

    # Alleen eerste 200 afbeeldingen gebruiken
    $images = $images | Select-Object -First 200

    # Split 80/20 voor train/val
    $train_count = [math]::Floor($images.Count * 0.8)
    $train_images = $images | Select-Object -First $train_count
    $val_images = $images | Select-Object -Skip $train_count

    # Verwerk training images
    foreach ($img in $train_images) {
        $img_name = "$letter" + "_" + $img.BaseName
        $target_img_path = Join-Path $output_dir "images\train\$img_name.jpg"
        Copy-Item $img.FullName $target_img_path

        $label_path = Join-Path $output_dir "labels\train\$img_name.txt"
        "$class_id 0.5 0.5 0.8 0.8" | Out-File -Encoding ascii $label_path
    }

    # Verwerk validation images
    foreach ($img in $val_images) {
        $img_name = "$letter" + "_" + $img.BaseName
        $target_img_path = Join-Path $output_dir "images\val\$img_name.jpg"
        Copy-Item $img.FullName $target_img_path

        $label_path = Join-Path $output_dir "labels\val\$img_name.txt"
        "$class_id 0.5 0.5 0.8 0.8" | Out-File -Encoding ascii $label_path
    }

    $class_id++
}
