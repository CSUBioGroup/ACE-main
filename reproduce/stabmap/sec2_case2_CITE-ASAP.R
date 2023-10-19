library(StabMap)
# library(SingleCellMultiModal)
library(scran)
library(Matrix)

set.seed(2021)

data_dir = "/home/yxh/data/DOGMA/"
out_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/stabmap/cite-asap/'

# loading 
cite_rna_lognorm = (Matrix::readMM(paste0(data_dir, 'CITE-seq/rna_mat_norm.mtx')))
cite_adt_lognorm = (Matrix::readMM(paste0(data_dir, 'CITE-seq/adt_mat_norm.mtx')))
asap_atac_lognorm = readMM(paste0(data_dir, 'ASAP-seq/atac_mat_norm.mtx'))
asap_adt_lognorm  = readMM(paste0(data_dir, 'ASAP-seq/adt_mat_norm.mtx'))

cite_cell_names  = read.table(file=paste0(data_dir, 'CITE-seq/cell_names.csv'), sep=',', header=T)
asap_cell_names  = read.table(file=paste0(data_dir, 'ASAP-seq/cell_names.csv'), sep=',', header=T)
cite_rna_names   = read.table(file=paste0(data_dir, 'CITE-seq/hvg_names.csv'), sep=',', header=T)
cite_adt_names   = read.table(file=paste0(data_dir, 'CITE-seq/adt_names.csv'), sep=',', header=T)
asap_atac_names  = read.table(file=paste0(data_dir, 'ASAP-seq/hvp_names.csv'), sep=',', header=T)
asap_adt_names   = read.table(file=paste0(data_dir, 'ASAP-seq/adt_names.csv'), sep=',', header=T)

cite_meta_data   = read.table(file=paste0(data_dir, 'CITE-seq/metadata.csv'), sep=',', header=T)
asap_meta_data   = read.table(file=paste0(data_dir, 'ASAP-seq/metadata.csv'), sep=',', header=T)

# set attr
rownames(cite_rna_lognorm) = cite_rna_names[,2]
rownames(cite_adt_lognorm) = cite_adt_names[,2]
rownames(asap_atac_lognorm) = asap_atac_names[,2]
rownames(asap_adt_lognorm) = asap_adt_names[,2]

colnames(cite_rna_lognorm) = cite_cell_names$x
colnames(cite_adt_lognorm) = cite_cell_names$x
colnames(asap_atac_lognorm) = asap_cell_names$x
colnames(asap_adt_lognorm) = asap_cell_names$x

table(cite_meta_data$batch)
b1_idx = grep('control', cite_meta_data$batch)
b2_idx = grep('stim', cite_meta_data$batch) 

b3_idx = grep('control', asap_meta_data$batch)
b4_idx = grep('stim', asap_meta_data$batch) 

b1_data = rbind(cite_rna_lognorm[, b1_idx], cite_adt_lognorm[, b1_idx])
b2_data = rbind(cite_rna_lognorm[, b2_idx], cite_adt_lognorm[, b2_idx])
b3_data = rbind(asap_atac_lognorm[, b3_idx], asap_adt_lognorm[, b3_idx])
b4_data = rbind(asap_atac_lognorm[, b4_idx], asap_adt_lognorm[, b4_idx])

# stabmap !!!!!!!!!!!!!!
assay_list_indirect = list(
  batch1 = as(b1_data, 'dgCMatrix'),    # dgCMatrix
  batch2 = as(b2_data, 'dgCMatrix'), # dgCMatrix
  batch3 = as(b3_data, 'dgCMatrix'),
  batch4 = as(b4_data, 'dgCMatrix')
)

lapply(assay_list_indirect, dim)

# Running !!!!!!!!!!
# return matrix array
stab_indirect = stabMap(assay_list_indirect,
                        reference_list = c("batch1"),
                        maxFeatures=60000,
                        plot = FALSE)
write.csv(as.data.frame(stab_indirect), 
  file=paste0(out_dir, 'embed2.csv'),
  quote=F, row.names=T)
