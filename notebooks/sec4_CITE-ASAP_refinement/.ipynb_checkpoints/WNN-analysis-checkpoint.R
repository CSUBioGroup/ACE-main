Sys.setenv("OMP_NUM_THREADS" = 4)
Sys.setenv("OPENBLAS_NUM_THREADS" = 4)
Sys.setenv("MKL_NUM_THREADS" = 6)
Sys.setenv("VECLIB_MAXIMUM_THREADS" = 4)
Sys.setenv("NUMEXPR_NUM_THREADS" = 6)

library(Seurat)
library(SeuratDisk)
library(Signac)
library(ggplot2)
library(cowplot)
library(dplyr)
# library(SeuratData)
library(hdf5r)
library(Matrix)
library(harmony)

# ====================================================================================
#                                     CITE
# ====================================================================================

data_dir = '../../data/DOGMA/'

coembed <- readRDS(paste0(data_dir, '22July2020_Seurat_Coembed4.rds'))
citeseq <- subset(coembed, orig.ident=='RNA')

# DefaultAssay(citeseq) <- "RNA"
# citeseq <- SCTransform(citeseq, verbose = FALSE)

# DefaultAssay(citeseq) <- 'SCT'
# citeseq <- FindVariableFeatures(citeseq, nfeatures=5000, assay='SCT')

# citeseq@meta.data[["celltypes"]] <- celltypes
citeseq@meta.data[['batch']] = paste0('citeseq_', citeseq@meta.data$stim)


#### official pipeline
DefaultAssay(citeseq) <- 'RNA'
citeseq <- NormalizeData(citeseq) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()
citeseq <- RunHarmony(citeseq, group.by.vars="batch", reduction = 'pca', reduction.save = "harmony_RNA")

DefaultAssay(citeseq) <- 'ADT'
# # we will use all ADT features for dimensional reduction
# # we set a dimensional reduction name to avoid overwriting the 
VariableFeatures(citeseq) <- rownames(citeseq[["ADT"]])
citeseq <- NormalizeData(citeseq, normalization.method = 'CLR', margin = 2) %>% 
  ScaleData() %>% RunPCA(reduction.name = 'apca')
citeseq <- RunHarmony(citeseq, group.by.vars="batch", reduction = 'apca', reduction.save = "harmony_ADT")

# Identify multimodal neighbors. These will be stored in the neighbors slot, 
# and can be accessed using bm[['weighted.nn']]
# The WNN graph can be accessed at bm[["wknn"]], 
# and the SNN graph used for clustering at bm[["wsnn"]]
# Cell-specific modality weights can be accessed at bm$RNA.weight
citeseq <- FindMultiModalNeighbors(
  citeseq, reduction.list = list("harmony_RNA", "harmony_ADT"), 
  dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
)

citeseq <- RunUMAP(citeseq, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
citeseq <- FindClusters(citeseq, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)

celltypes <- rep('0', length(citeseq@meta.data$seurat_clusters))
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(9))] <- "Myeloid"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(10))] <- "Myeloid"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(4))] <- "B"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(3))] <- "NK"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(2,5))] <- "T"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(0,1,6,7,8,11))] <- "T"
citeseq@meta.data[["celltypes"]] <- celltypes

cite_wnn_umap_df = data.frame(Embeddings(object=citeseq, reduction="wnn.umap"))
rownames(cite_wnn_umap_df) = colnames(citeseq)
write.csv(cite_wnn_umap_df, 
	file='./citeseq_wnn.umap.csv', 
	quote=F, row.names=T)

# ====================================================================================
#                                     ASAP
# ====================================================================================

asapseq <- subset(coembed, orig.ident=='ATAC')
cdf <- readRDS(paste0(data_dir, "ASAP_embedding_CLRadt.rds"))  
control_cells <- gsub("Control#", "", rownames(cdf)[cdf$sample == "Control"])
stim_cells <- gsub("Stim#", "", rownames(cdf)[cdf$sample == "Stim"])

pbmc <- readRDS(paste0(data_dir, 'pbmc_LLL_processed.rds'))
DefaultAssay(pbmc) <- 'peaks'
frags.stim <- CreateFragmentObject(
  path = paste0(data_dir, "stim_fragments.tsv.gz"),
  cells = stim_cells
)
stim.counts <- FeatureMatrix(
  fragments = frags.stim,
  features = granges(pbmc),
  cells = stim_cells
)
frags.control <- CreateFragmentObject(
  path = paste0(data_dir, "control_fragments.tsv.gz"),
  cells = control_cells
)
control.counts <- FeatureMatrix(
  fragments = frags.control,
  features = granges(pbmc),
  cells = control_cells
)
asap_cells <- c(control_cells, stim_cells)
# shared_cell <- asap_cells[asap_cells %in% colnames(atac_mat)]
peaks <- cbind(control.counts, stim.counts)
colnames(peaks) <- colnames(asapseq[['ADT']]@counts)
asapseq@meta.data$cellids <- colnames(peaks)
asapseq[["peaks"]] <- CreateChromatinAssay(
    counts = peaks,
    sep = c(":", "-"),
    #genome = 'hg38',
    #fragments = '../../../asap_large_data_files/multiome_pbmc_stim/input/fragments.tsv.gz',
    min.cells = 0,
    min.features = 0
)
asapseq@meta.data$batch = asapseq@meta.data$stim

DefaultAssay(asapseq) <- 'ADT'
VariableFeatures(asapseq) <- rownames(asapseq[["ADT"]])
asapseq <- NormalizeData(asapseq, normalization.method = 'CLR', margin = 2) %>% 
  					ScaleData() %>% RunPCA(reduction.name = 'apca')
asapseq <- RunHarmony(asapseq, group.by.vars="batch", reduction='apca', reduction.save = "harmony_ADT", project.dim = F)

DefaultAssay(asapseq) <- "peaks"
asapseq <- RunTFIDF(asapseq)
asapseq <- FindTopFeatures(asapseq, min.cutoff = 'q50')
asapseq <- RunSVD(asapseq)
asapseq <- RunHarmony(asapseq, group.by.vars="batch", reduction = 'lsi', reduction.save = "harmony_ATAC", project.dim = F)

asapseq <- FindMultiModalNeighbors(asapseq, reduction.list = list("harmony_ADT", "harmony_ATAC"), dims.list = list(1:50, 2:50))
asapseq <- RunUMAP(asapseq, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")

asap_wnn_umap_df = data.frame(Embeddings(object=asapseq, reduction="wnn.umap"))
rownames(asap_wnn_umap_df) = colnames(asapseq)
write.csv(asap_wnn_umap_df, 
	file='./asapseq_wnn.umap.csv', 
	quote=F, row.names=T)