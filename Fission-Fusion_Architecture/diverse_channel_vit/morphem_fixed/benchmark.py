import pandas as pd
import os
import json
from morphem_fixed.evaluation import evaluate, create_umap

import warnings
warnings.filterwarnings("ignore")

def save_results(results, dest_dir, dataset, classifier, knn_metric):
    # Helper function
    # Save results for each dataset as a json dictionary at dest_dir
    full_reports_dict = {}
    full_reports_dict['target_encoding'] = results["encoded_target"]
    for task_ind, task in enumerate(results["tasks"]):
        full_reports_dict[task] = results["reports_dict"][task_ind]

    if not os.path.exists(dest_dir+ '/'):
        os.makedirs(dest_dir+ '/')
    
    if classifier == 'knn':
        dict_path = f'{dest_dir}/{dataset}_{classifier}_{knn_metric}_results.json'
    else:
        dict_path = f'{dest_dir}/{dataset}_{classifier}_results.json'
        
    with open(dict_path, 'w') as f:
        json.dump(full_reports_dict, f)

    return
            
def run_benchmark(root_dir, dest_dir, feature_dir, feature_file, classifier='knn', umap=False, use_gpu=True, knn_metric='l2', features_path=None, datasets=None): #adding for subset tests 4/9

    print(f"Running benchmark with the following parameters:")
    print(f"root_dir: {root_dir}")
    print(f"dest_dir: {dest_dir}")
    print(f"feature_dir: {feature_dir}")
    print(f"feature_file: {feature_file}")
    print(f"classifier: {classifier}")
    print(f"datasets: {datasets}")
    
    # Check if directories exist
    print(f"Checking if directories exist:")
    print(f"root_dir exists: {os.path.exists(root_dir)}")
    print(f"dest_dir exists: {os.path.exists(dest_dir)}")
    print(f"feature_dir exists: {os.path.exists(feature_dir)}")
    
    # encode dataset, task, and classifier
        # If not provided, default to all 4/9
    if datasets is None:
        datasets = ['Allen', 'HPA', 'CP']

    # task_dict = pd.DataFrame({'dataset':['Allen', 'HPA', 'CP'], 
    #                           'classifier':[classifier for i in range(3)], \
    #                           'leave_out': [None, 'Task_three', 'Task_four'], \
    #                           'leaveout_label': [None, 'cell_type', 'Plate'], \
    #                           'umap_label': ['Structure', 'cell_type', 'source'] 
    #                          })
    #new logic 4/9
    if datasets is not None:
        datasets_to_run = datasets
    else:
        datasets_to_run = ['Allen', 'HPA', 'CP']

    task_dict = pd.DataFrame({
        'dataset': datasets_to_run,
        'classifier': [classifier for _ in datasets_to_run],
        'leave_out': [None if d == 'Allen' else ('Task_three' if d == 'HPA' else 'Task_four') for d in datasets_to_run],
        'leaveout_label': [None if d == 'Allen' else ('cell_type' if d == 'HPA' else 'Plate') for d in datasets_to_run],
        'umap_label': ['Structure' if d == 'Allen' else ('cell_type' if d == 'HPA' else 'source') for d in datasets_to_run],
    })

    full_result_df = pd.DataFrame(columns=['dataset', 'task', 'classifier', 'accuracy', 'f1_score_macro'])
    
    # Iterrate over each dataset
    for idx, row in task_dict.iterrows():
        dataset        = row.dataset
        classifier     = row.classifier
        leave_out      = row.leave_out
        leaveout_label = row.leaveout_label
        umap_label     = row.umap_label
        
        # Use dataset-specific features path if features_path is not explicitly provided
        # This ensures each dataset uses its own features file
        current_features_path = features_path
        if current_features_path is None:
            current_features_path = f'{feature_dir}/{dataset}/{feature_file}'
        else:
            # If a specific features_path was provided but we're evaluating multiple datasets,
            # try to find dataset-specific features
            if len(datasets_to_run) > 1:
                dataset_specific_path = f'{feature_dir}/{dataset}/{feature_file}'
                if os.path.exists(dataset_specific_path):
                    current_features_path = dataset_specific_path
                    print(f"Using dataset-specific features for {dataset}: {current_features_path}")
                else:
                    print(f"Warning: Using the same features file for all datasets. This may cause index mismatches.")
                    print(f"Consider generating separate feature files for each dataset.")

        df_path = f'{root_dir}/{dataset}/enriched_meta.csv'
        
        print(f"Dataset: {dataset}")
        print(f"features_path: {current_features_path}")
        print(f"df_path: {df_path}")
        
        # Skip this dataset if the metadata or features file doesn't exist
        if not os.path.exists(df_path):
            print(f"Warning: Metadata file not found: {df_path}")
            print(f"Skipping evaluation for dataset: {dataset}")
            continue
            
        if not os.path.exists(current_features_path):
            print(f"Warning: Features file not found: {current_features_path}")
            print(f"Skipping evaluation for dataset: {dataset}")
            continue
        
        # Create umap and run classification
        if umap:
            create_umap(dataset, 
                        current_features_path, 
                        df_path, 
                        dest_dir, 
                        ['Label', umap_label])
            
        results = evaluate(current_features_path, 
                           df_path, 
                           leave_out, 
                           leaveout_label, 
                           classifier, 
                           use_gpu, 
                           knn_metric)

        # Print the full results
        print('Results:')
        for task_ind, task in enumerate(results["tasks"]):
            print(f'Results for {dataset} {task} with {classifier} :')
            print(results["reports_str"][task_ind])
        
        # Save results as dictionary
        save_results(results, dest_dir, dataset, classifier, knn_metric)
        
        # Save results as csv
        result_temp = pd.DataFrame({'dataset': [dataset for i in range(len(results["tasks"]))],\
                        'task': results["tasks"],'classifier': [classifier for i in range(len(results["tasks"]))],\
                        'accuracy': results["accuracies"],'f1_score_macro': results["f1scores_macro"]})
        full_result_df = pd.concat([full_result_df, result_temp]).reset_index(drop=True)
    
    if classifier == 'knn':        
        full_result_df.to_csv(f'{dest_dir}/{classifier}_{knn_metric}_full_results.csv', index=False) 
    else:
        full_result_df.to_csv(f'{dest_dir}/{classifier}_full_results.csv', index=False)
        
    return full_result_df
