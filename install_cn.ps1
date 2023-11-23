Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1

if (!(Test-Path -Path "venv")) {
    Write-Output  "Creating venv for python..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "Installing deps..."
pip install -U -r requirements.txt -i https://mirror.baidu.com/pypi/simple
pip install -e generative-models -i https://mirror.baidu.com/pypi/simple
pip install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
pip install gradio -i https://mirror.baidu.com/pypi/simple

Write-Output "Install completed"
Read-Host | Out-Null ;
