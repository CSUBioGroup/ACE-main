library(Seurat)
library(SeuratDisk)
library(SingleCellExperiment)
library(Matrix)
library(hdf5r)
library(dplyr)


# =========================================
# save CITE reference
# download from: https://zenodo.org/records/7779017#.ZCMojezMJqs
cite = readRDS('/home/cb213/local/cache3/yxh/Data/seurat4-CITE-reference/pbmc_multimodal_2023.rds')

file.h5 <- H5File$new("../../data/COVID-19/Bridge/cite.h5", mode = "w")
file.h5[['cellID']] = colnames(cite)
file.h5[["RNA.shape"]] <- cite[['SCT']]@data@Dim
file.h5[["RNA.norm.data"]] <- cite[['SCT']]@data@x
file.h5[["RNA.norm.indices"]] <- cite[['SCT']]@data@i
file.h5[["RNA.norm.indptr"]] <- cite[['SCT']]@data@p
file.h5[["RNA.count.data"]] <- cite[['SCT']]@counts@x
file.h5[["RNA.count.indices"]] <- cite[['SCT']]@counts@i
file.h5[["RNA.count.indptr"]] <- cite[['SCT']]@counts@p
file.h5[['rna_names']] = rownames(cite[['SCT']])

file.h5[["adt_norm_data"]] <- cite[['ADT']]@data
file.h5[["adt_count_data"]] <- as.matrix(cite[['ADT']]@counts)
file.h5[['protein_names']] = rownames(cite[['ADT']])
file.h5[["umap"]] = as.matrix(Embeddings(cite, 'wnn.umap'))
# metadata
file.h5[['donor']] = cite@meta.data$donor
file.h5[['celltype.l1']] = cite@meta.data$celltype.l1
file.h5[['celltype.l2']] = cite@meta.data$celltype.l2
file.h5[['celltype.l3']] = cite@meta.data$celltype.l3
file.h5[['Phase']] = cite@meta.data$Phase
file.h5[['X_index']] = cite@meta.data$X_index
file.h5[['lane']] = cite@meta.data$lane
file.h5[['time']] = cite@meta.data$time

file.h5$close_all()