library(StabMap)
# library(SingleCellMultiModal)
library(scran)
library(Matrix)

set.seed(2021)

data_dir = "/home/yxh/data/Seurat_demo_data/bm_cite/"
out_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/stabmap/bm-cite-donorSplit/'

# loading 
rna_lognorm = (Matrix::readMM(paste0(data_dir, 'rna_mat_norm.mtx')))
adt_lognorm = (Matrix::readMM(paste0(data_dir, 'adt_mat_norm.mtx')))
cell_names  = read.table(file=paste0(data_dir, 'cell_names.csv'), sep=',', header=T)
rna_names   = read.table(file=paste0(data_dir, 'gene_names.csv'), sep=',', header=T)
adt_names   = read.table(file=paste0(data_dir, 'adt_names.csv'), sep=',', header=T)
meta_data   = read.table(file=paste0(data_dir, 'metadata.csv'), sep=',', header=T)

# train_idx = read.table(file=paste0(data_dir, 'train_idx.csv'), sep=',', header=T)
# test_idx  = read.table(file=paste0(data_dir, 'test_idx.csv'), sep=',', header=T)
train_idx = grep(T, meta_data$donor=='batch1')
test_idx  = grep(T, meta_data$donor=='batch2')
hvg_names = read.table(file=paste0(data_dir, 'hvg_names.csv'), sep=',', header=T)  # exported from seurat

# split
mult_rna_lognorm = rna_lognorm[, train_idx]
mult_adt_lognorm = adt_lognorm[, train_idx]
single_rna_lognorm = rna_lognorm[, test_idx]
single_adt_lognorm = adt_lognorm[, test_idx]

# set attr
rownames(mult_rna_lognorm) = rna_names$x
rownames(single_rna_lognorm) = rna_names$x
rownames(mult_adt_lognorm) = adt_names$x
rownames(single_adt_lognorm) = adt_names$x

colnames(mult_rna_lognorm) = cell_names$x[train_idx]
colnames(mult_adt_lognorm) = cell_names$x[train_idx]
colnames(single_rna_lognorm) = paste0('gex_', cell_names$x[test_idx])
colnames(single_adt_lognorm) = paste0('adt_', cell_names$x[test_idx])

# subsetting hvgs
mult_rna_lognorm = mult_rna_lognorm[hvg_names[,2], ]
single_rna_lognorm = single_rna_lognorm[hvg_names[,2], ]

mult_lognorm = rbind(mult_rna_lognorm, mult_adt_lognorm)  # logcount(): dgCMatrix
dim(mult_lognorm)

# stabmap !!!!!!!!!!!!!!
assay_list_indirect = list(
  RNA = as(single_rna_lognorm, 'dgCMatrix'),    # dgCMatrix
  Multiome = mult_lognorm, # dgCMatrix
  ADT = as(single_adt_lognorm, 'dgCMatrix')   # dgCMatrix
)

lapply(assay_list_indirect, dim)

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/BM-CITE-donorSplit-mosaic-overlapFeature.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mosaicDataUpSet(assay_list_indirect, plot = FALSE)
dev.off()

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/BM-CITE-donorSplit-mosaic-topology.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mdt_indirect = mosaicDataTopology(assay_list_indirect)
plot(mdt_indirect)
dev.off()

# Running !!!!!!!!!!
# return matrix array
stab_indirect = stabMap(assay_list_indirect,
                        reference_list = c("Multiome"),
                        maxFeatures=20000,
                        plot = FALSE)
write.csv(as.data.frame(stab_indirect), 
  file=paste0(out_dir, 'embed.csv'),
  quote=F, row.names=T)


