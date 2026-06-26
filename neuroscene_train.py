import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

# --- OpenScene Imports ---
from run.distill import get_model as get_openscene_model
from dataset.feature_loader import FusedFeatureLoader, collation_fn_eval_all
from util import config
from run.evaluate import precompute_text_related_properties # Reusing their text setup

# --- NeuroScene Imports ---
from neuroscene.bixt_fusion_model import NeuroSceneBiXTFusion # Your custom BiXT file
import wandb


def get_parser():
    parser = argparse.ArgumentParser(description='NeuroScene Hybrid Training')
    parser.add_argument('--config', type=str, default='config/scannet/eval_openseg.yaml', help='config file')
    parser.add_argument('--save_path', type=str, default='./checkpoints/neuroscene_v1.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--distill_weight', type=float, default=0.5, help='Lambda weight for open-vocabulary distillation')
    parser.add_argument('--temperature', type=float, default=0.07, help='InfoNCE temperature')
    
    # Allows overriding config via command line
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def main():
    args = get_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Initializing NeuroScene Training Pipeline ---")

    # 1. Precompute Ground-Truth Text Embeddings (InfoNCE Targets)
    labelset_name = args.data_root.split('/')[-1] if not hasattr(args, 'labelset') else args.labelset
    text_features, labelset, mapper, palette = precompute_text_related_properties(labelset_name)
    text_features = text_features.to(device).float()
    
    # 2. Load the OpenScene 3D Backbone (TEACHER / EXTRACTOR)
    print("Loading OpenScene Backbone...")
    openscene_model = get_openscene_model(args).to(device)
    
    # FREEZE the OpenScene Backbone completely
    for param in openscene_model.parameters():
        param.requires_grad = False
    openscene_model.eval()

    # 3. Initialize NeuroScene BiXT Fusion Model (STUDENT / FUSER)
    print("Initializing BiXT Fusion Module...")
    neuroscene_model = NeuroSceneBiXTFusion(d_model=512, num_latents=128, num_layers=3).to(device)
    neuroscene_model.train()

    # 4. Optimizer & Loss Config
    optimizer = torch.optim.AdamW(neuroscene_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # CrossEntropy natively ignores the 255 (class-agnostic) points during InfoNCE calculation
    criterion_infonce = torch.nn.CrossEntropyLoss(ignore_index=255)

    # 5. Dataloader Setup
    train_data = FusedFeatureLoader(
        datapath_prefix=args.data_root,
        datapath_prefix_feat=args.data_root_2d_fused_feature,
        voxel_size=args.voxel_size, 
        split='train', # Ensure this points to the training split
        aug=True,      # Enable data augmentation for robust geometric learning
        eval_all=True, 
        input_color=False
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.train_workers, collate_fn=collation_fn_eval_all, pin_memory=True
    )

    # Initialize W&B for logging
    print("Initializing W&B...")
    wandb.init(
        project="NeuroScene-Framework",
        name="BiXT-Hybrid-Loss-Run-1",
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "distill_weight_initial": args.distill_weight,
            "architecture": "BiXT-Dual-Modality",
            "dataset": "ScanNet200"
        }
    )

    # 6. The Core Training Loop
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        epoch_loss_infonce = 0.0
        epoch_loss_distill = 0.0
        
        # We use a dynamic learning weight scheduler. It starts relying heavily on the 
        # visual distillation to stabilize features, and slowly lowers it to trust InfoNCE more.
        current_lambda = args.distill_weight * (1.0 - (epoch / args.epochs))

        progress_bar = tqdm(train_loader, desc="Training")
        for coords, feat, label, feat_3d, mask, inds_reverse in progress_bar:
            optimizer.zero_grad()
            
            label = label.to(device, non_blocking=True)
            valid_mask = mask[inds_reverse]

            # A. Extract Frozen Backbone Features
            with torch.no_grad():
                from MinkowskiEngine import SparseTensor # Imported here to avoid global init issues
                sinput = SparseTensor(feat.to(device, non_blocking=True), coords.to(device, non_blocking=True))
                
                # Spatial Geometry Features [N, 512]
                spatial_feats = openscene_model(sinput)[inds_reverse, :].float()
                
                # Visual Pixel Features (Ground Truth Oracle) [N, 512]
                visual_feats = feat_3d.to(device, non_blocking=True)[inds_reverse, :].float()

            # B. Forward Pass through BiXT
            spatial_feats = spatial_feats.unsqueeze(0) # Add batch dim: [1, N, 512]
            visual_feats = visual_feats.unsqueeze(0)   # [1, N, 512]
            
            updated_spatial = neuroscene_model(spatial_feats, visual_feats) 
            updated_spatial = updated_spatial.squeeze(0) # Remove batch dim: [N, 512]

            # Filter out points that fall outside the camera frustum completely
            if valid_mask.sum() == 0:
                continue
                
            updated_spatial = updated_spatial[valid_mask]
            visual_feats_flat = visual_feats.squeeze(0)[valid_mask]
            label_valid = label[valid_mask]

            # C. Loss 1: InfoNCE (Semantic Language Grounding for Known Classes)
            updated_spatial_norm = F.normalize(updated_spatial, p=2, dim=-1)
            text_features_norm = F.normalize(text_features, p=2, dim=-1)
            
            logits = (updated_spatial_norm @ text_features_norm.t()) / args.temperature
            loss_infonce = criterion_infonce(logits, label_valid)

            # D. Loss 2: Open-Vocabulary Distillation (Preserving unlabelled knowledge)
            unlabeled_mask = (label_valid == 255)
            if unlabeled_mask.sum() > 0:
                cosine_sim = F.cosine_similarity(
                    updated_spatial[unlabeled_mask], 
                    visual_feats_flat[unlabeled_mask], 
                    dim=-1
                )
                # Minimize distance (1 - Cosine Similarity)
                loss_distill = (1.0 - cosine_sim).mean() 
            else:
                loss_distill = torch.tensor(0.0).to(device)

            # E. Compute Hybrid Objective and Backpropagate
            total_loss = loss_infonce + (current_lambda * loss_distill)
            
            total_loss.backward()
            optimizer.step()

            # Logging
            epoch_loss_infonce += loss_infonce.item()
            epoch_loss_distill += loss_distill.item()
            progress_bar.set_postfix({'InfoNCE': f"{loss_infonce.item():.4f}", 'Distill': f"{loss_distill.item():.4f}"})

            wandb.log({
                "Train/Total_Loss": total_loss.item(),
                "Train/InfoNCE_Loss": loss_infonce.item(),
                "Train/Distillation_Loss": loss_distill.item(),
                "Hyperparameters/Current_Lambda": current_lambda,
                "Hyperparameters/Learning_Rate": optimizer.param_groups[0]['lr']
            })

        # Save Checkpoint at the end of the epoch
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': neuroscene_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.save_path)
        print(f"Epoch {epoch+1} Complete. Checkpoint saved to {args.save_path}")

    # Close the W&B run
    wandb.finish()

if __name__ == '__main__':
    main()