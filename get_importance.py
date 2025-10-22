import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import argparse
import json

def load_data(model_name):
    with open(f'./head_results/training_labels.npy', "rb") as f:
        train_labels = np.load(f)
    with open(f'./head_results/test_labels.npy', "rb") as f:
        test_labels = np.load(f)
    # with open(f'./head_results/{model_name}_cot_qa_head_wise_train_0.pkl', 'rb') as f:
    #     train_data_0 = pickle.load(f)
    # with open(f'./head_results/{model_name}_cot_qa_head_wise_train_1.pkl', 'rb') as f:
    #     train_data_1 = pickle.load(f)
    # with open(f'./head_results/{model_name}_cot_qa_head_wise_train_2.pkl', 'rb') as f:
    #     train_data_2 = pickle.load(f)
    # with open(f'./head_results/{model_name}_cot_qa_head_wise_train_3.pkl', 'rb') as f:
    #     train_data_3 = pickle.load(f)
    # train_data = train_data_0 + train_data_1 + train_data_2 + train_data_3
    with open(f'./head_results/{model_name}_cot_qa_head_wise_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(f'./head_results/{model_name}_cot_qa_head_wise_test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open(f"./head_results/{model_name}_cot_qa_token_positions_train.pkl", 'rb') as f:
        train_position = pickle.load(f)
    with open(f"./head_results/{model_name}_cot_qa_token_positions_test.pkl", 'rb') as f:
        test_position = pickle.load(f)
    with open(f'./head_results/{model_name}_cot_qa_topk_position_train.pkl', "rb") as f:
        train_topk_position = pickle.load(f)
    with open(f'./head_results/{model_name}_cot_qa_topk_position_test.pkl', "rb") as f:
        test_topk_position = pickle.load(f)
    print(len(train_data))
    return train_data, test_data, train_labels, test_labels, train_position, test_position, train_topk_position, test_topk_position


def get_head_weight(train, test, layer_num, heads_num, dim, position, head_num, layer_bias,
                    train_position, test_position, train_topk_position, test_topk_position):
    def extract_features(data, position_data, topk_data, is_train=True):
        features = []
        for i in range(len(data)):
            token_num = data[i].shape[1]
            #print("token_num", token_num)
            #print("topk_data", len(topk_data))
            token_select = 0
            if position == "first":
                token_select = 0
            elif position == "last":
                token_select = token_num - 1
            elif position == "meaning":
                token_select = position_data[i][0]
            elif position == "topk":
                token_select = [x for x in topk_data[i] if 0 <= x < token_num]
            elif position == "full":
                token_select = [x for x in position_data[i] if 0 <= x < token_num]

            reshaped = data[i].reshape(layer_num, token_num, heads_num, dim)
            reshaped = reshaped[:, token_select, :, :]

            if position in ["topk", "full"]:
                reshaped = np.mean(reshaped, axis=1)

            if layer_bias:
                layer_means = reshaped.mean(axis=1)
                expanded = np.expand_dims(layer_means, axis=1).repeat(heads_num, axis=1)
                reshaped = np.concatenate([reshaped, expanded], axis=-1)
                reshaped = reshaped.reshape(layer_num * heads_num, dim * 2)
            else:
                reshaped = reshaped.reshape(layer_num * heads_num, dim)

            features.append(reshaped if head_num == -1 else reshaped[head_num])
        return np.array(features)

    train_feat = extract_features(train, train_position, train_topk_position)
    test_feat = extract_features(test, test_position, test_topk_position, is_train=False)
    return train_feat, test_feat


class FeatureLevelClassifier(nn.Module):
    def __init__(self, input_dim, num_features=1024, num_classes=12):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, 64)
        self.classifier = nn.Sequential(
            nn.Linear(64 * num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.feature_proj(x)
        return self.classifier(x.view(x.size(0), -1))


def train_model(model, train_loader, val_loader, device, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # k = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            #print(xb, yb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            # k = k + 1
            # print(f"Loss: {loss.item()}, data {k}")
            # if torch.isnan(loss):
            #     print("Loss is NaN, data {k}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Acc: {acc:.4f}")
        
def find_elbow_point(y):
    x = np.arange(len(y))
    y = np.array(y)

    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)
    
    vecs = np.stack([x - x[0], y - y[0]], axis=1)
    proj_lengths = np.dot(vecs, line_vec)
    proj_points = np.outer(proj_lengths, line_vec)
    dists = np.linalg.norm(vecs - proj_points, axis=1)

    elbow_index = np.argmax(dists)
    return elbow_index
    
def compute_feature_importance_per_label(model, dataset, num_features=1024, num_labels=12):
    model.eval()
    model.feature_proj.requires_grad_(True) 

    importance_matrix = torch.zeros((num_labels, num_features))  # (6, 1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for label in range(num_labels):
        grads_list = []
        acts_list = []
        count = 0
        
        for xb, yb in DataLoader(dataset, batch_size=16):
            xb = xb.to(device)
            yb = yb.to(device)
            # print(xb, yb)
            mask = (yb == label)
            if mask.sum() == 0:
                continue

            xb_sel = xb[mask]
            xb_sel.requires_grad = True
            # 前向传播
            proj = model.feature_proj(xb_sel)  # shape: (B, 1024, 64)
            proj.retain_grad()

            output = model.classifier(proj.view(proj.size(0), -1))  # (B, num_classes)
            logit = output[:, label].sum()  

            # 反向传播
            model.zero_grad()
            logit.backward(retain_graph=True)

            # 梯度 × 激活
            grad = proj.grad.detach()             # (B, 1024, 64)
            act = proj.detach()                   # (B, 1024, 64)
            gxa = (grad * act).mean(dim=2)        # (B, 1024)

            grads_list.append(gxa.cpu())
            count += gxa.size(0)

            # if count >= 100:  
            #     break

        if grads_list:
            importance_matrix[label] = torch.cat(grads_list, dim=0).mean(dim=0)

    return importance_matrix  # shape: (num_labels, 1024)

def main(args):
    train_data, test_data, train_labels, test_labels, train_pos, test_pos, topk_train, topk_test = load_data(args.model)
    empty_indices = [i for i, item in enumerate(train_pos) if item == []]
    if empty_indices:
        print(f"Empty indices in train_pos: {empty_indices}")
        train_data = [i for j, i in enumerate(train_data) if j not in empty_indices]
        train_labels = np.delete(train_labels, empty_indices)
        train_pos = [i for j, i in enumerate(train_pos) if j not in empty_indices]
        topk_train = [i for j, i in enumerate(topk_train) if j not in empty_indices]
    encoder = LabelEncoder()
    y_train = torch.tensor(encoder.fit_transform(train_labels), dtype=torch.long)
    y_test = torch.tensor(encoder.transform(test_labels), dtype=torch.long)
    label_mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
    label_names = []
    for i in label_mapping:
        label_names.append(i)
    for bias in [True]:
        for pos in ['topk']:
            print(f"\nRunning with BIAS={bias}, POSITION={pos}")
            X_train, X_test = get_head_weight(train_data, test_data, args.layer_num, args.heads_num, args.dim,
                                              pos, -1, bias, train_pos, test_pos, topk_train, topk_test)
            input_dim = args.dim * 2 if bias else args.dim
            print(X_test)
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1, shuffle=False)
            val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = FeatureLevelClassifier(input_dim=input_dim, num_features=args.layer_num * args.heads_num).to(device)
            train_model(model, train_loader, val_loader, device)
            dataset = TensorDataset(X_test, y_test)
            importance_matrix = compute_feature_importance_per_label(
                model=model,
                dataset=dataset,
                num_features=args.layer_num * args.heads_num,
                num_labels=len(encoder.classes_)
            )
            importance_matrix = importance_matrix.cpu().numpy() if hasattr(importance_matrix, 'cpu') else importance_matrix
            num_labels, num_features = importance_matrix.shape
            sorted_indices = {}
            for label in range(num_labels):
                importances = importance_matrix[label].tolist()
                
                sorted_indices[label_names[label]] = sorted(enumerate(importances), key=lambda x: abs(x[1]), reverse=True)
                
            # 保存 importance_matrix
            if not os.path.exists(f"./main_results/{args.model}"):
                os.makedirs(f"./main_results/{args.model}")
            with open(f"./main_results/{args.model}/layer_{bias}_position_{pos}.json","w") as f:
                json.dump(sorted_indices,f)
            elbow = {}
            for label in range(num_labels):
                importances = importance_matrix[label].tolist()
                
                sorted_indices[label_names[label]] = sorted(enumerate(importances), key=lambda x: abs(x[1]), reverse=True)
                elbow[label_names[label]] = find_elbow_point([abs(i[1]) for i in sorted_indices[label_names[label]]]).tolist()
            with open(f"./main_results/{args.model}/layer_{bias}_position_{pos}_elbow.json","w") as f:
                json.dump(elbow,f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yi-1.5-6B')
    parser.add_argument('--layer_num', type=int, default=32)
    parser.add_argument('--heads_num', type=int, default=32)
    parser.add_argument('--dim', type=int, default=128)
    args = parser.parse_args()
    main(args)