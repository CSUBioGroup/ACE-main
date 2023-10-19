library(StabMap)
library(glue)
library(scran)
library(Matrix)
library(rlang)
library(Seurat)
library(Signac)

set.seed(2021)

data_dir = "/home/yxh/data/DOGMA/"
data_dir2= '/home/yxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case4/dogma/'
out_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/stabmap/case4_dogma/'

# loading 
rna_count = (Matrix::readMM(paste0(data_dir, 'RNA/rna_mat_count.mtx')))
adt_count = (Matrix::readMM(paste0(data_dir, 'ADT/adt_mat_count.mtx')))
atac_count = (Matrix::readMM(paste0(data_dir, 'ATAC/atac_mat_count.mtx')))
rna_names   = read.table(file=paste0(data_dir, 'RNA/hvg_names.csv'), sep=',', header=T)
adt_names   = read.table(file=paste0(data_dir, 'ADT/adt_names.csv'), sep=',', header=T)
atac_names   = read.table(file=paste0(data_dir, 'ATAC/hvp_names.csv'), sep=',', header=T)

cell_names  = read.table(file=paste0(data_dir, 'cell_names.csv'), sep=',', header=T)
meta_data   = read.table(file=paste0(data_dir, 'metadata.csv'), sep=',', header=T)

# train_idx = read.table(file=paste0(data_dir, 'train_idx.csv'), sep=',', header=T)
# test_idx  = read.table(file=paste0(data_dir, 'test_idx.csv'), sep=',', header=T)
test_idx  = grep(T, meta_data$stim == 'Stim')

for (p in c(0.1, 0.2, 0.4, 0.8)){
  for (r in c(0, 1, 2)){
      train_idx = read.table(paste0(data_dir2, glue('p={p}_r={r}_new_train_idx.csv')), sep=',', header=T)$X0 + 1
      test_rna_idx = read.table(paste0(data_dir2, glue('p={p}_r={r}_test_rna_idx.csv')), sep=',', header=T)$X0 + 1
      test_adt_idx = read.table(paste0(data_dir2, glue('p={p}_r={r}_test_adt_idx.csv')), sep=',', header=T)$X0 + 1
      test_atac_idx = read.table(paste0(data_dir2, glue('p={p}_r={r}_test_atac_idx.csv')), sep=',', header=T)$X0 + 1
    
      # split
      mult_rna_count = rna_count[, train_idx]
      mult_adt_count = adt_count[, train_idx]
      mult_atac_count = atac_count[, train_idx]
      single_rna_count = rna_count[, test_rna_idx]
      single_adt_count = adt_count[, test_adt_idx]
      single_atac_count = atac_count[, test_atac_idx]

      # set attr
      rownames(mult_rna_count) = rna_names[,2]
      rownames(single_rna_count) = rna_names[,2]
      rownames(mult_adt_count) = adt_names[,2]
      rownames(single_adt_count) = adt_names[,2]
      rownames(mult_atac_count) = atac_names[,2]
      rownames(single_atac_count) = atac_names[,2]

      colnames(mult_rna_count) = cell_names$x[train_idx]
      colnames(mult_adt_count) = cell_names$x[train_idx]
      colnames(mult_atac_count) = cell_names$x[train_idx]
      colnames(single_rna_count) = paste0('gex_', cell_names$x[test_rna_idx])
      colnames(single_adt_count) = paste0('adt_', cell_names$x[test_adt_idx])
      colnames(single_atac_count) = paste0('atac_', cell_names$x[test_atac_idx])

      n_mult = dim(mult_rna_count)[2]
      n_single_rna = dim(single_rna_count)[2]
      n_single_adt = dim(single_adt_count)[2]
      n_single_atac = dim(single_atac_count)[2]

      # select features
      data_rna <- CreateSeuratObject(counts = cbind(mult_rna_count, single_rna_count), project = "dogma")
      data_adt <- CreateSeuratObject(counts = cbind(mult_adt_count, single_adt_count))
      data_atac <- CreateSeuratObject(counts = cbind(mult_atac_count, single_atac_count))

      VariableFeatures(data_rna) <- rownames(data_rna)
      data_rna <- NormalizeData(data_rna)
      # variable_rna_features = VariableFeatures(data_rna)

      VariableFeatures(data_adt) <- rownames(data_adt)
      data_adt <- NormalizeData(data_adt, normalization.method = 'CLR', margin = 2)
      # variable_adt_features = VariableFeatures(data_adt)

      data_atac <- RunTFIDF(data_atac)
      data_atac <- FindTopFeatures(data_atac, min.cutoff = "q60")
      variable_atac_features = VariableFeatures(data_atac)
      print(length(variable_atac_features))

      mult_lognorm = rbind(
        data_rna[['RNA']]@data[, 1:n_mult], 
        data_adt[['RNA']]@data[, 1:n_mult],  # (n_mult+1):(n_mult+n_single)
        GetAssayData(data_atac)[variable_atac_features, 1:n_mult]
      )  
      dim(mult_lognorm)

      single_rna_lognorm = as(data_rna[['RNA']]@data[, (n_mult+1):(n_mult+n_single_rna)], 'dgCMatrix')
      single_adt_lognorm = as(data_adt[['RNA']]@data[, (n_mult+1):(n_mult+n_single_adt)], 'dgCMatrix') 
      single_atac_lognorm = as(GetAssayData(data_atac)[variable_atac_features, (n_mult+1):(n_mult+n_single_atac)], 'dgCMatrix') 

      # stabmap !!!!!!!!!!!!!!
      assay_list_indirect = list(
        RNA = as(single_rna_lognorm, 'dgCMatrix'),    # dgCMatrix
        Multiome = mult_lognorm, # dgCMatrix
        ADT = as(single_adt_lognorm, 'dgCMatrix'),   # dgCMatrix
        ATAC=as(single_atac_lognorm, 'dgCMatrix')
      )

      lapply(assay_list_indirect, dim)

      # jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/DOGMA-StimSplit-mosaic-overlapFeature.jpg", 
      #   width = 4, height = 4, units = 'in', res = 300)
      mosaicDataUpSet(assay_list_indirect, plot = FALSE)
      # dev.off()

      # jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/DOGMA-StimSplit-mosaic-topology.jpg", 
      #   width = 4, height = 4, units = 'in', res = 300)
      mdt_indirect = mosaicDataTopology(assay_list_indirect)
      # plot(mdt_indirect)
      # dev.off()

      # Running !!!!!!!!!!
      # return matrix array
      stab_indirect = stabMap(assay_list_indirect,
                              reference_list = c("Multiome"),
                              maxFeatures=60000,
                              plot = FALSE)
      write.csv(as.data.frame(stab_indirect), 
        file=paste0(out_dir, glue('p={p}_r={r}_embed.csv')),
        quote=F, row.names=T)
  }
}
