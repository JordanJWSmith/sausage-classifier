$files = Get-ChildItem ".\input\valid\sausage"

for ($i=0; $i -lt $files.Count; $i++) {
    $outfile = $files[$i].FullName
    $filename = Split-Path $outfile -Leaf
    python inference.py --input=input/valid/sausage/"$filename" --display=False
}