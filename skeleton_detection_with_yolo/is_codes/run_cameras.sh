#!/bin/bash


# Rodar todos em background e guardar os PIDs
#python3 stream.py 1 &
#PID1=$!

#python3 stream.py 2 &
#PID2=$!

#python3 stream.py 3 &
#PID3=$!

#python3 stream.py 4 &
#PID4=$!

# Função para matar tudo quando receber CTRL+C
#trap "echo 'Parando...'; kill PID1 PID2 PID3 PID4; exit" SIGINT SIGHUP

# Espera os processos terminarem
#wait

# Permitir job control
set -m

# Rodar todos em background
python3 stream.py 1 &
python3 stream.py 2 &
python3 stream.py 3 &
python3 stream.py 4 &

# Captura sinais (Ctrl+C ou fechar terminal)
trap "echo 'Parando...'; kill 0; exit" SIGINT SIGHUP

# Espera os processos
wait
