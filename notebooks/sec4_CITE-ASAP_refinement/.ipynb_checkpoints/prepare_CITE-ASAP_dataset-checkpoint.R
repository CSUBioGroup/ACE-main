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

DefaultAssay(citeseq) <- "RNA"
citeseq <- SCTransform(citeseq, verbose = FALSE)

DefaultAssay(citeseq) <- 'SCT'
citeseq <- FindVariableFeatures(citeseq, nfeatures=5000, assay='SCT')
RNA <- citeseq@assays$SCT@counts[VariableFeatures(citeseq),]
# RNA <- RNA[apply(RNA>0, 1, sum)>=250,]
ADT <- citeseq@assays$ADT@counts
# ADT <- ADT[apply(ADT>0, 1, sum)>=250,]
cat(dim(RNA), dim(ADT))

celltypes <- rep('0', length(citeseq@meta.data$seurat_clusters))
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(9))] <- "Mono"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(10))] <- "DC"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(4))] <- "B"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(3))] <- "NK"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(2,5))] <- "CD8 T"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(0,1,6,7,8,11))] <- "CD4 T"
citeseq@meta.data[["celltypes"]] <- celltypes
citeseq@meta.data[['batch']] = paste0('citeseq_', citeseq@meta.data$stim)

# to_dir = '/home/yxh/data/DOGMA/CITE-seq/'
# writeMM(as(RNA, 'dgCMatrix'), paste0(to_dir, 'rna_mat_count.mtx'))
# write.csv(rownames(RNA), file=paste0(to_dir, 'hvg_names.csv'), quote=F, row.names=T)
# writeMM(as(ADT, 'dgCMatrix'), paste0(to_dir, 'adt_mat_count.mtx'))
# write.csv(rownames(ADT), file=paste0(to_dir, 'adt_names.csv'), quote=F, row.names=T)

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
DefaultAssay(asapseq) <- 'peaks'
asapseq <- FindTopFeatures(asapseq, min.cutoff = 'q25')

# ===================================================
#              analyze, CITE+ASAP
# ===================================================

# rna
DefaultAssay(citeseq) = 'SCT'
cite_rna_count = citeseq@assays$SCT@counts
CITE_rna = CreateSeuratObject(cite_rna_count)
CITE_rna@meta.data$batch = citeseq@meta.data$stim
CITE_rna <- CITE_rna  %>% 
  NormalizeData() %>% 
  FindVariableFeatures() %>%
  ScaleData() %>%
  RunPCA(verbose = FALSE, assay = "RNA", reduction.name = "pca") %>%
  RunHarmony( group.by.vars = 'batch', reduction = 'pca', assay.use = 'RNA',
             project.dim = FALSE,  reduction.save = "harmony_RNA")

to_dir = '../../data/DOGMA/CITE-seq/'
writeMM(as(GetAssayData(CITE_rna)[VariableFeatures(CITE_rna),], 'dgCMatrix'), paste0(to_dir, 'rna_mat_norm.mtx'))
writeMM(as(GetAssayData(CITE_rna, slot='counts')[VariableFeatures(CITE_rna),], 'dgCMatrix'), paste0(to_dir, 'rna_mat_count.mtx'))
writeMM(as(Embeddings(object=CITE_rna, reduction="harmony_RNA"), 'dgCMatrix'), paste0(to_dir, 'rna_harmonypca.mtx'))
write.csv(as.data.frame(VariableFeatures(CITE_rna)), file=paste0(to_dir, 'hvg_names.csv'), quote=F, row.names=T)

celltypes <- rep('0', length(citeseq@meta.data$seurat_clusters))
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(9))] <- "Mono"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(10))] <- "DC"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(4))] <- "B"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(3))] <- "NK"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(2,5))] <- "CD8 T"
celltypes[citeseq@meta.data$seurat_clusters %in% as.character(c(0,1,6,7,8,11))] <- "CD4 T"
CITE_rna@meta.data[["celltypes"]] <- celltypes

write.csv(colnames(CITE_rna), file=paste0(to_dir, 'cell_names.csv'), quote=F, row.names=T)
write.csv(CITE_rna@meta.data, file=paste0(to_dir, 'metadata.csv'), quote=F, row.names=T)

# atac
asap_atac_count = asapseq@assays$peaks@counts
ASAP_atac = CreateSeuratObject(asap_atac_count, assay='ATAC')
ASAP_atac@meta.data$batch = asapseq@meta.data$stim
ASAP_atac <- RunTFIDF(ASAP_atac) %>% 
  FindTopFeatures(min.cutoff = 'q25') %>%
  RunSVD() %>%
  RunHarmony( group.by.vars = 'batch', reduction = 'lsi', assay.use = 'peaks',
             project.dim = FALSE,  reduction.save = "harmony_Peaks")

to_dir = '../../data/DOGMA/ASAP-seq/'
writeMM(as(GetAssayData(ASAP_atac)[VariableFeatures(ASAP_atac),], 'dgCMatrix'), paste0(to_dir, 'atac_mat_norm.mtx'))
writeMM(as(GetAssayData(ASAP_atac, slot='counts')[VariableFeatures(ASAP_atac),], 'dgCMatrix'), paste0(to_dir, 'atac_mat_count.mtx'))
writeMM(as(Embeddings(object=ASAP_atac, reduction="harmony_Peaks"), 'dgCMatrix'), paste0(to_dir, 'atac_harmonypca.mtx'))
write.csv(as.data.frame(VariableFeatures(ASAP_atac)), file=paste0(to_dir, 'hvp_names.csv'), quote=F, row.names=T)

celltypes <- rep('0', length(asapseq@meta.data$seurat_clusters))
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(9))] <- "Mono"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(10))] <- "DC"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(4))] <- "B"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(3))] <- "NK"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(2,5))] <- "CD8 T"
celltypes[asapseq@meta.data$seurat_clusters %in% as.character(c(0,1,6,7,8,11))] <- "CD4 T"
ASAP_atac@meta.data[["celltypes"]] <- celltypes

write.csv(colnames(ASAP_atac), file=paste0(to_dir, 'cell_names.csv'), quote=F, row.names=T)
write.csv(ASAP_atac@meta.data, file=paste0(to_dir, 'metadata.csv'), quote=F, row.names=T)

# adt
cite_adt_count = citeseq@assays$ADT@counts
asap_adt_count = asapseq@assays$ADT@counts
inter_adt_names = intersect(rownames(cite_adt_count), rownames(asap_adt_count))
cite_adt_count = cite_adt_count[inter_adt_names,]
asap_adt_count = asap_adt_count[inter_adt_names,]
cite_asap_adt_count = cbind(cite_adt_count, asap_adt_count)
CITE_ASAP_adt = CreateSeuratObject(cite_asap_adt_count, assay='ADT')
CITE_ASAP_adt@meta.data$stim = c(citeseq@meta.data$stim, asapseq@meta.data$stim)
CITE_ASAP_adt@meta.data$seq  = c(rep('cite', dim(citeseq)[2]), rep('asap', dim(asapseq)[2]))
CITE_ASAP_adt <- CITE_ASAP_adt  %>% 
  NormalizeData(assay='ADT', normalization.method = "CLR", margin = 2) %>%
  FindVariableFeatures(assay = "ADT") %>% 
  ScaleData(assay = "ADT", do.scale = FALSE) %>%
  RunPCA(verbose = FALSE, assay = "ADT", reduction.name = 'apca') %>%
  RunHarmony( group.by.vars = c('stim', 'seq'), reduction = 'apca', assay.use = 'ADT',
             project.dim = FALSE,  reduction.save = "harmony_ADT")

n_cite = dim(citeseq)[2] 
n_asap = dim(asapseq)[2]
to_dir1 = '../../data/DOGMA/CITE-seq/'
writeMM(as(GetAssayData(CITE_ASAP_adt)[VariableFeatures(CITE_ASAP_adt),1:n_cite], 'dgCMatrix'), paste0(to_dir1, 'adt_mat_norm.mtx'))
writeMM(as(GetAssayData(CITE_ASAP_adt, slot='counts')[VariableFeatures(CITE_ASAP_adt), 1:n_cite], 'dgCMatrix'), paste0(to_dir1, 'adt_mat_count.mtx'))
writeMM(as(Embeddings(object=CITE_ASAP_adt, reduction="harmony_ADT")[1:n_cite, ], 'dgCMatrix'), paste0(to_dir1, 'adt_harmonypca.mtx'))
write.csv(as.data.frame(VariableFeatures(CITE_ASAP_adt)), file=paste0(to_dir1, 'adt_names.csv'), quote=F, row.names=T)

to_dir2 = '../../data/DOGMA/ASAP-seq/'
writeMM(as(GetAssayData(CITE_ASAP_adt)[VariableFeatures(CITE_ASAP_adt),(n_cite+1):(n_cite+n_asap)], 'dgCMatrix'), paste0(to_dir2, 'adt_mat_norm.mtx'))
writeMM(as(GetAssayData(CITE_ASAP_adt, slot='counts')[VariableFeatures(CITE_ASAP_adt), (n_cite+1):(n_cite+n_asap)], 'dgCMatrix'), paste0(to_dir2, 'adt_mat_count.mtx'))
writeMM(as(Embeddings(object=CITE_ASAP_adt, reduction="harmony_ADT")[(n_cite+1):(n_cite+n_asap), ], 'dgCMatrix'), paste0(to_dir2, 'adt_harmonypca.mtx'))
write.csv(as.data.frame(VariableFeatures(CITE_ASAP_adt)), file=paste0(to_dir, 'adt_names.csv'), quote=F, row.names=T)
