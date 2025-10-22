for model in "llama3.2_3B_instruct"
do
    for function in "Retrieval"
    do   
        for head in randomk
        do
            for token in topk
            do
            python ablation_topk.py \
            --function_name "$function" \
            --use_head "$head" \
            --model_name "$model" \
            --token_use "$token" \
            --output_dir "head_acc.csv"
            done
        done
    done
done