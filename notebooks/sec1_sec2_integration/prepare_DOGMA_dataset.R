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
library(harmony)
library(hdf5r)
library(Matrix)


############################################################
#
# DOGMA-seq
#
############################################################
data_dir = '../../data/DOGMA/'
pbmc <- readRDS(paste0(data_dir, 'pbmc_LLL_processed.rds'))

DefaultAssay(pbmc) <- 'SCT'
# pbmc <- FindVariableFeatures(pbmc, nfeatures=5000, assay='SCT')
RNA <- pbmc@assays$SCT@counts#[VariableFeatures(pbmc),]
# RNA <- RNA[apply(RNA>0, 1, sum)>=500,]
ADT <- pbmc@assays$ADT@counts
# ADT <- ADT[apply(ADT>0, 1, sum)>=500,]
DefaultAssay(pbmc) <- 'peaks'
# pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q25')
peaks <- pbmc@assays$peaks@counts#[VariableFeatures(pbmc),]
# peaks <- peaks[apply(peaks>0, 1, sum)>=500,]             
peaks <- peaks[!(startsWith(rownames(peaks), 'chrX') | 
                 startsWith(rownames(peaks), 'chrY')),]
peaks <- peaks[rownames(peaks)[order(match(rownames(peaks),rownames(pbmc)))],]

colnames(RNA) 
colnames(ADT)
colnames(peaks)

identical(colnames(RNA), colnames(ADT))
identical(colnames(RNA), colnames(peaks))

# library(hdf5r)
# file.h5 <- H5File$new(paste0(data_dir, "DOGMA_pbmc.h5"), mode = "w")
# file.h5[["ADT"]] <- as.matrix(ADT)

# file.h5[["RNA.shape"]] <- RNA@Dim
# file.h5[["RNA.data"]] <- RNA@x
# file.h5[["RNA.indices"]] <- RNA@i
# file.h5[["RNA.indptr"]] <- RNA@p

# file.h5[["peaks.shape"]] <- peaks@Dim
# file.h5[["peaks.data"]] <- peaks@x
# file.h5[["peaks.indices"]] <- peaks@i
# file.h5[["peaks.indptr"]] <- peaks@p

# file.h5[["cell_ids"]] <- colnames(RNA)
# file.h5[["gene_names"]] <- rownames(RNA)
# file.h5[["ADT_names"]] <- rownames(ADT)
# file.h5[["peak_names"]] <- rownames(peaks)
# file.h5[["wsnn_res.0.2"]] <- as.character(pbmc@meta.data$wsnn_res.0.2)
# file.h5[["wsnn_res.0.8"]] <- as.character(pbmc@meta.data$wsnn_res.0.8)
# file.h5[["predicted.celltypes.l1"]] <- pbmc@meta.data$predicted.celltype.l1
# file.h5[["predicted.celltypes.l2"]] <- pbmc@meta.data$predicted.celltype.l2
# file.h5[["predicted.celltypes.l1.score"]] <- pbmc@meta.data$predicted.celltype.l1.score
# file.h5[["predicted.celltypes.l2.score"]] <- pbmc@meta.data$predicted.celltype.l2.score

# file.h5[["batches"]] <- pbmc@meta.data$stim
# file.h5$close_all()


############################################################
#
# Preprocessing
#
############################################################

stim <- pbmc@meta.data$stim   # batch label
data_ref <- CreateSeuratObject(RNA)
data_ref[["ADT"]] <- CreateAssayObject(ADT, assay='ADT')
data_ref[['peaks']] <- CreateChromatinAssay(peaks)
data_ref@meta.data$stim <- stim

DefaultAssay(data_ref) <- "RNA"
data_ref <- data_ref  %>% 
  NormalizeData() %>% 
  FindVariableFeatures() %>%
  ScaleData() %>%
  RunPCA(verbose = FALSE, assay = "RNA", reduction.name = "pca") %>%
  RunHarmony( group.by.vars = 'stim', reduction = 'pca', assay.use = 'RNA',
             project.dim = FALSE,  reduction.save = "harmony_RNA")

# LSI dim reduction
DefaultAssay(data_ref) <- "peaks"
data_ref <- RunTFIDF(data_ref) %>% 
  FindTopFeatures(min.cutoff = 'q25') %>%
  RunSVD() %>%
  RunHarmony( group.by.vars = 'stim', reduction = 'lsi', assay.use = 'peaks',
             project.dim = FALSE,  reduction.save = "harmony_Peaks")

# Do it for ADT
DefaultAssay(data_ref) <- "ADT"
data_ref <- data_ref  %>% 
  NormalizeData(assay = "ADT", normalization.method = "CLR", margin = 2) %>%
  FindVariableFeatures(assay = "ADT") %>% 
  ScaleData(assay = "ADT", do.scale = FALSE) %>%
  RunPCA(verbose = FALSE, assay = "ADT", reduction.name = 'apca') %>%
  RunHarmony( group.by.vars = 'stim', reduction = 'apca', assay.use = 'ADT',
             project.dim = FALSE,  reduction.save = "harmony_ADT")

#### export
to_dir = '../../data/DOGMA/'
DefaultAssay(data_ref) <- 'RNA'
writeMM(as(GetAssayData(data_ref)[VariableFeatures(data_ref),], 'dgCMatrix'), paste0(to_dir, 'rna_mat_norm.mtx'))
writeMM(as(GetAssayData(data_ref, slot='counts')[VariableFeatures(data_ref),], 'dgCMatrix'), paste0(to_dir, 'rna_mat_count.mtx'))
writeMM(as(Embeddings(object=data_ref, reduction="harmony_RNA"), 'dgCMatrix'), paste0(to_dir, 'rna_harmonypca.mtx'))
write.csv(as.data.frame(VariableFeatures(data_ref)), file=paste0(to_dir, 'hvg_names.csv'), quote=F, row.names=T)

DefaultAssay(data_ref) <- 'ADT'
writeMM(as(GetAssayData(data_ref)[VariableFeatures(data_ref),], 'dgCMatrix'), paste0(to_dir, 'adt_mat_norm.mtx'))
writeMM(as(GetAssayData(data_ref, slot='counts')[VariableFeatures(data_ref),], 'dgCMatrix'), paste0(to_dir, 'adt_mat_count.mtx'))
writeMM(as(Embeddings(object=data_ref, reduction="harmony_ADT"), 'dgCMatrix'), paste0(to_dir, 'adt_harmonypca.mtx'))
write.csv(as.data.frame(VariableFeatures(data_ref)), file=paste0(to_dir, 'adt_names.csv'), quote=F, row.names=T)

DefaultAssay(data_ref) <- 'peaks'
writeMM(as(GetAssayData(data_ref)[VariableFeatures(data_ref),], 'dgCMatrix'), paste0(to_dir, 'atac_mat_norm.mtx'))
writeMM(as(GetAssayData(data_ref, slot='counts')[VariableFeatures(data_ref),], 'dgCMatrix'), paste0(to_dir, 'atac_mat_count.mtx'))
writeMM(as(Embeddings(object=data_ref, reduction="harmony_Peaks"), 'dgCMatrix'), paste0(to_dir, 'atac_harmonypca.mtx'))
write.csv(as.data.frame(VariableFeatures(data_ref)), file=paste0(to_dir, 'hvp_names.csv'), quote=F, row.names=T)

write.csv(colnames(data_ref), file=paste0(to_dir, 'cell_names.csv'), quote=F, row.names=T)
write.csv(pbmc@meta.data, file=paste0(to_dir, 'metadata.csv'), quote=F, row.names=T)

