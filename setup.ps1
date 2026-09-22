$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"
$Requirements = Join-Path $ProjectRoot "requirements.txt"

$PythonExecutable = $null
$PythonArguments = @()

$PythonLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    $PythonExecutable = $PythonLauncher.Source
    $PythonArguments = @("-3")
}
else {
    $PythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        try {
            & $PythonCommand.Source --version | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $PythonExecutable = $PythonCommand.Source
            }
        }
        catch {
            $PythonExecutable = $null
        }
    }
}

if (-not $PythonExecutable) {
    Write-Host "Python ainda não está instalado ou não está disponível no terminal." -ForegroundColor Yellow
    Write-Host "Instale o Python 3.12 ou mais recente pelo site python.org." -ForegroundColor Yellow
    Write-Host "Durante a instalação, marque a opção 'Add python.exe to PATH'." -ForegroundColor Yellow
    Write-Host "Depois, reinicie o VS Code e execute esta tarefa novamente." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $VirtualEnvironment)) {
    Write-Host "Criando o ambiente isolado do projeto..." -ForegroundColor Cyan
    & $PythonExecutable @PythonArguments -m venv $VirtualEnvironment
    if ($LASTEXITCODE -ne 0) {
        throw "Não foi possível criar o ambiente virtual."
    }
}

$EnvironmentPython = Join-Path $VirtualEnvironment "Scripts\python.exe"

Write-Host "Instalando as dependências..." -ForegroundColor Cyan
& $EnvironmentPython -m pip install --upgrade pip
& $EnvironmentPython -m pip install -r $Requirements

Write-Host "Ambiente preparado com sucesso." -ForegroundColor Green
Write-Host "Agora execute a tarefa '2. Extrair políticas dos PDFs'." -ForegroundColor Green
