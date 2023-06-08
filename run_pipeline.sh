#!/bin/bash

show_pipeline_selection_menu() {
    echo "Choose the pipeline to execute:"
    echo "1. Approach 1 : Pre-finetuning -> Finetuning"
    echo "2. Approach 2 : Finetuning"
    echo "3. Exit"
    echo
}

show_quantize_menu() {
    echo "Choose whether to quantize models or not:"
    echo "1. Yes"
    echo "2. No"
    echo
}

option1() {
    show_quantize_menu
    read -p "Enter your choice: " quantize_choice
    cd src 
    python preprocess.py
    python run_task.py --task ss
    python run_task.py --task ss-ke
    case $quantize_choice in
        1)
            python quantize.py --task ss-ke
            echo "Finished..."
            break
            ;;
        2)
            echo "Finished..."
            break
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
    
}

option2() {
    show_quantize_menu
    read -p "Enter your choice: " quantize_choice
    cd src
    python run_task.py --task ke
    case $quantize_choice in
        1)
            python quantize.py --task ke
            echo "Finished..."
            break
            ;;
        2)
            echo "Finished..."
            break
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
}

while true; do
    show_pipeline_selection_menu
    read -p "Enter your choice: " choice
    case $choice in
        1)
            option1
            ;;
        2)
            option2
            ;;
        3)
            echo "Exiting..."
            break
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
    echo
done