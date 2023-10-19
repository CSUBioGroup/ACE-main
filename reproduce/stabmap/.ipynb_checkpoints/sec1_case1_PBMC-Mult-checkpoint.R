library(StabMap)
# library(SingleCellMultiModal)
library(scran)
library(Matrix)

set.seed(2021)

data_dir = "/home/yxh/data/Seurat_demo_data/pbmc_multiome/"
out_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/stabmap/pbmc_mult/'

# loading 
rna_lognorm = (Matrix::readMM(paste0(data_dir, 'rna_mat_norm.mtx')))
atac_lognorm = (Matrix::readMM(paste0(data_dir, 'atac_mat_norm.mtx')))
cell_names  = read.table(file=paste0(data_dir, 'cell_names.csv'), sep=',', header=T)
rna_names   = read.table(file=paste0(data_dir, 'gene_names.csv'), sep=',', header=T)
atac_names   = read.table(file=paste0(data_dir, 'atac_names.csv'), sep=',', header=T)
meta_data   = read.table(file=paste0(data_dir, 'metadata.csv'), sep=',', header=T)

train_idx = read.table(file=paste0(data_dir, 'train_idx.csv'), sep=',', header=T)
test_idx  = read.table(file=paste0(data_dir, 'test_idx.csv'), sep=',', header=T)
hvg_names = read.table(file=paste0(data_dir, 'hvg_names.csv'), sep=',', header=T)  # exported from scanpy.flavor='seurat_v3'
hvp_names = read.table(file=paste0(data_dir, 'hvp_names.csv'), sep=',', header=T)  # exported from scanpy.flavor='seurat_v3'

# split
mult_rna_lognorm = rna_lognorm[, train_idx$X0+1]
mult_atac_lognorm = atac_lognorm[, train_idx$X0+1]
single_rna_lognorm = rna_lognorm[, test_idx$X0+1]
single_atac_lognorm = atac_lognorm[, test_idx$X0+1]

# set attr
rownames(mult_rna_lognorm) = rna_names$x
rownames(single_rna_lognorm) = rna_names$x
rownames(mult_atac_lognorm) = atac_names$x
rownames(single_atac_lognorm) = atac_names$x

colnames(mult_rna_lognorm) = cell_names$x[train_idx$X0+1]
colnames(mult_atac_lognorm) = cell_names$x[train_idx$X0+1]
colnames(single_rna_lognorm) = paste0('gex_', cell_names$x[test_idx$X0+1])
colnames(single_atac_lognorm) = paste0('atac_', cell_names$x[test_idx$X0+1])

# subsetting hvgs
mult_rna_lognorm = mult_rna_lognorm[hvg_names$X0, ]
single_rna_lognorm = single_rna_lognorm[hvg_names$X0, ]
mult_atac_lognorm = mult_atac_lognorm[hvp_names$X0, ]
single_atac_lognorm = single_atac_lognorm[hvp_names$X0, ]

mult_lognorm = rbind(mult_rna_lognorm, mult_atac_lognorm)  # logcount(): dgCMatrix
dim(mult_lognorm)

# stabmap !!!!!!!!!!!!!!
assay_list_indirect = list(
  RNA = as(single_rna_lognorm, 'dgCMatrix'),    # dgCMatrix
  Multiome = mult_lognorm, # dgCMatrix
  ATAC = as(single_atac_lognorm, 'dgCMatrix')   # dgCMatrix
)

lapply(assay_list_indirect, dim)

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/PBMC-Mult-mosaic-overlapFeature.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mosaicDataUpSet(assay_list_indirect, plot = FALSE)
dev.off()

jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/PBMC-Mult-mosaic-topology.jpg", 
  width = 4, height = 4, units = 'in', res = 300)
mdt_indirect = mosaicDataTopology(assay_list_indirect)
plot(mdt_indirect)
dev.off()

# Running !!!!!!!!!!
# return matrix array
stab_indirect = stabMap(assay_list_indirect,
                        reference_list = c("Multiome"),
                        maxFeatures=30000,
                        plot = FALSE)
write.csv(as.data.frame(stab_indirect), 
  file=paste0(out_dir, 'embed.csv'),
  quote=F, row.names=T)


