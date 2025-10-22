for low in "Retrieval" "Knowledge Recall" "Semantic Understanding" "Syntactic Understanding"
do   
    for high in "Math Calculation" "Induction" "Inference" "Logical Reasoning" "Decision-making"
    do
        for head in topk
        do
            python ablation_cot.py \
            --low_function_name "$low" \
            --high_function_name "$high" \
            --use_head "$head" 
        done
    done
done