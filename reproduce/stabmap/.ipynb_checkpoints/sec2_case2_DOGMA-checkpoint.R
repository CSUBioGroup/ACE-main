library(StabMap)
# library(SingleCellMultiModal)
library(scran)
library(Matrix)

set.seed(2021)

data_dir = "/home/yxh/data/DOGMA/"
out_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/stabmap/dogma-StimSplit/'

# loading 
rna_lognorm = (Matrix::readMM(paste0(data_dir, 'RNA/rna_mat_norm.mtx')))
adt_lognorm = (Matrix::readMM(paste0(data_dir, 'ADT/adt_mat_norm.mtx')))
atac_lognorm = (Matrix::readMM(paste0(data_dir, 'ATAC/atac_mat_norm.mtx')))
rna_names   = read.table(file=paste0(data_dir, 'RNA/hvg_names.csv'), sep=',', header=T)
adt_names   = read.table(file=paste0(data_dir, 'ADT/adt_names.csv'), sep=',', header=T)
atac_names   = read.table(file=paste0(data_dir, 'ATAC/hvp_names.csv'), sep=',', header=T)

cell_names  = read.table(file=paste0(data_dir, 'cell_names.csv'), sep=',', header=T)
meta_data   = read.table(file=paste0(data_dir, 'metadata.csv'), sep=',', header=T)

# train_idx = read.table(file=paste0(data_dir, 'train_idx.csv'), sep=',', header=T)
# test_idx  = read.table(file=paste0(data_dir, 'test_idx.csv'), sep=',', header=T)
train_idx = grep(T, meta_data$stim == 'Control')
test_idx  = grep(T, meta_data$stim == 'Stim')

# split
mult_rna_lognorm = rna_lognorm[, train_idx]
mult_adt_lognorm = adt_lognorm[, train_idx]
mult_atac_lognorm = atac_lognorm[, train_idx]
single_rna_lognorm = rna_lognorm[, test_idx]
single_adt_lognorm = adt_lognorm[, test_idx]
single_atac_lognorm = atac_lognorm[, test_idx]

# set attr
rownames(mult_rna_lognorm) = rna_names[,2]
rownames(single_rna_lognorm) = rna_names[,2]
rownames(mult_adt_lognorm) = adt_names[,2]
rownames(single_adt_lognorm) = adt_names[,2]
rownames(mult_atac_lognorm) = atac_names[,2]
rownames(single_atac_lognorm) = atac_names[,2]

colnames(mult_rna_lognorm) = cell_names$x[train_idx]
colnames(mult_adt_lognorm) = cell_names$x[train_idx]
colnames(mult_atac_lognorm) = cell_names$x[train_idx]
colnames(single_rna_lognorm) = paste0('gex_', cell_names$x[test_idx])
colnames(single_adt_lognorm) = paste0('adt_', cell_names$x[test_idx])
colnames(single_atac_lognorm) = paste0('atac_', cell_names$x[test_idx])


mult_lognorm = rbind(mult_rna_lognorm, mult_adt_lognorm, mult_atac_lognorm)  # logcount(): dgCMatrix
dim(mult_lognorm)

# stabmap !!!!!!!!!!!!!!
assay_list_indirect = list(
  RNA = as(single_rna_lognorm, 'dgCMatrix'),    # dgCMatrix
  Multiome = mult_lognorm, # dgCMatrix
  ADT = as(single_adt_lognorm, 'dgCMatrix'),   # dgCMatrix
  ATAC=as(single_atac_lognorm, 'dgCMatrix')
)

lapply(assay_list_indirect, dim)

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/DOGMA-StimSplit-mosaic-overlapFeature.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mosaicDataUpSet(assay_list_indirect, plot = FALSE)
dev.off()

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/DOGMA-StimSplit-mosaic-topology.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mdt_indirect = mosaicDataTopology(assay_list_indirect)
plot(mdt_indirect)
dev.off()

# Running !!!!!!!!!!
# return matrix array
stab_indirect = stabMap(assay_list_indirect,
                        reference_list = c("Multiome"),
                        maxFeatures=60000,
                        plot = FALSE)
write.csv(as.data.frame(stab_indirect), 
  file=paste0(out_dir, 'embed.csv'),
  quote=F, row.names=T)


