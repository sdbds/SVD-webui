$model_path="./checkpoints/svd_xt.safetensors"
$outputs="./outputs"
$port=7860

Set-Location $PSScriptRoot
.\venv\Scripts\activate

$Env:HF_HOME = "./huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()

if ($port -ne 7860) {
  [void]$ext_args.Add("--port=$port")
}

python.exe webui.py `
--model_path=$model_path `
--outputs=$outputs $ext_args
