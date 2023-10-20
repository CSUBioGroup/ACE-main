library(Seurat)
library(SeuratDisk)
library(SingleCellExperiment)
library(Matrix)
library(hdf5r)
library(dplyr)

# download from: https://zenodo.org/record/5139561
obj2 = readRDS('/home/cb213/local/cache3/yxh/Data/COVID-2-CYTOF/CBD-KEY-CYTOF-WB-D/sceobj_depleted.RDS')
colnames(colData(obj2))
# [1] "sample_id"       "condition"       "patient_id"      "batch"          
# [5] "cellID"          "COMBAT_ID_Time"  "CyTOF_priority"  "major_cell_type"
# [9] "fine_cluster_id"

cov_idx = grep("COVID|HV", colData(obj2)$condition)
length(unique(colData(obj2)$COMBAT_ID_Time[cov_idx]))

sel_idx = c()
sample_list = unique(colData(obj2)$COMBAT_ID_Time)
for (smp in sample_list){
  smp_idx = grep(smp, colData(obj2)$COMBAT_ID_Time)
  smp_idx = intersect(cov_idx, smp_idx)
  if (length(smp_idx) >= 1000){
    sub_smp_idx = sample(smp_idx, 1000, replace=F)
    sel_idx = c(sel_idx, sub_smp_idx)
  }
  else{
    print(smp)
    print(length(smp_idx))
  }
}

length(sel_idx)  # 116000 samples

file.h5 <- H5File$new("../../data/CYTOF/sel_data.h5", mode = "w")
file.h5[['sel_idx']] = sel_idx
file.h5[["norm_data"]] <- as.matrix(assays(obj2)$exprs[, sel_idx])
file.h5[["umap"]] = as.matrix(reducedDims(obj2)$UMAP[sel_idx, ])
# metadata
file.h5[['sample_id']] = colData(obj2)$sample_id[sel_idx]	
file.h5[['condition']] = colData(obj2)$condition[sel_idx]
file.h5[['patient_id']] = colData(obj2)$patient_id[sel_idx]
file.h5[['batch']] = colData(obj2)$batch[sel_idx]
file.h5[['cellID']] = colData(obj2)$cellID[sel_idx]
file.h5[['COMBAT_ID_Time']] = colData(obj2)$COMBAT_ID_Time[sel_idx]
file.h5[['CyTOF_priority']] = colData(obj2)$CyTOF_priority[sel_idx]
file.h5[['major_cell_type']] = colData(obj2)$major_cell_type[sel_idx]
file.h5[['fine_cluster_id']] = colData(obj2)$fine_cluster_id[sel_idx]

file.h5[["protein_names"]] <- rownames(obj2)
file.h5$close_all()