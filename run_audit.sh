for random_state in {1..3}
do
    for dataset in "synthetic" "natural"
    do
        for model in "bm25" "sparta" "splade" "ance" "sbert" "use-qa" "colbert"
        do
            for emb in "sentence-t5-xl" #"nli-roberta-large" #"all-MiniLM-L6-v2"
            do
                echo "CUDA_VISIBLE_DEVICES=0 python -m src.experiments.run_audit --dataset'$dataset' --fit_dataset "synthetic" --model '$model' --embedding_model '$emb' --results_dir 'results_$emb' --random_state $random_state --spurious_dataset 'spurious'"
                CUDA_VISIBLE_DEVICES=1 python -m src.experiments.run_audit --dataset $dataset --fit_dataset "synthetic" --model "$model" --embedding_model "$emb" --results_dir "results_$emb" --random_state $random_state
            done
        done
    done
done