library(StabMap)
library(glue)
library(scran)
library(Matrix)
library(rlang)
library(Seurat)

set.seed(2021)

data_dir = "/home/yxh/data/Seurat_demo_data/bm_cite/"
data_dir2= '/home/yxh/gitrepo/multi-omics-matching/Senst_exp_inputs/case3/'
out_dir = '/home/yxh/gitrepo/multi-omics-matching/neurips2021_multimodal_topmethods-main/output/pretrain/stabmap/case3_bm-cite/'

# loading 
rna_count_mat = Matrix::readMM(paste0(data_dir, 'rna_mat_count.mtx'))
adt_count_mat = Matrix::readMM(paste0(data_dir, 'adt_mat_count.mtx'))

gene_names = read.table(file=paste0(data_dir, 'gene_names.csv'), sep=',', header=T)
protein_names = read.table(file=paste0(data_dir, 'adt_names.csv'), sep=',', header=T)
cell_names = read.table(file=paste0(data_dir, 'cell_names.csv'), sep=',', header=T)

rownames(rna_count_mat) = gene_names$x
rownames(adt_count_mat) = protein_names$x
colnames(rna_count_mat) = cell_names$x
colnames(adt_count_mat) = cell_names$x
  
for (p in c(0.1, 0.2, 0.4, 0.8)){
  for (r in c(0, 1, 2)){
    print('===================================')
    print(p)
    print(r)
    print('===================================')
    train_idx = read.table(paste0(data_dir2, glue('p={p}_r={r}_new_train_idx.csv')), sep=',', header=T)$X0 + 1
    test_rna_idx = read.table(paste0(data_dir2, glue('p={p}_r={r}_test_rna_idx.csv')), sep=',', header=T)$X0 + 1
    test_adt_idx = read.table(paste0(data_dir2, glue('p={p}_r={r}_test_adt_idx.csv')), sep=',', header=T)$X0 + 1

    mult_rna_count = rna_count_mat[, train_idx]
    mult_adt_count = adt_count_mat[, train_idx]
    single_rna_count = rna_count_mat[, test_rna_idx]
    single_adt_count = adt_count_mat[, test_adt_idx]

    n_mult = dim(mult_rna_count)[2]
    n_single_rna = dim(single_rna_count)[2]
    n_single_adt = dim(single_adt_count)[2]

    # select features
    data_rna <- CreateSeuratObject(counts = cbind(mult_rna_count, single_rna_count), project = "bm-cite-rna")
    data_adt <- CreateSeuratObject(counts = cbind(mult_adt_count, single_adt_count), project = "bm-cite-adt")

    data_rna <- NormalizeData(data_rna) %>% FindVariableFeatures()
    variable_rna_features = VariableFeatures(data_rna)
    VariableFeatures(data_adt) <- rownames(data_adt)
    data_adt <- NormalizeData(data_adt, normalization.method = 'CLR', margin = 2)
    variable_adt_features = VariableFeatures(data_adt)

    mult_lognorm = rbind(
      data_rna[['RNA']]@data[variable_rna_features, 1:n_mult], 
      data_adt[['RNA']]@data[variable_adt_features, 1:n_mult]  # (n_mult+1):(n_mult+n_single)
    )  
    dim(mult_lognorm)

    # stabmap !!!!!!!!!!!!!!
    single_rna_lognorm = as(data_rna[['RNA']]@data[variable_rna_features, (n_mult+1):(n_mult+n_single_rna)], 'dgCMatrix')
    single_adt_lognorm = as(data_adt[['RNA']]@data[variable_adt_features, (n_mult+1):(n_mult+n_single_adt)], 'dgCMatrix') 
    colnames(single_rna_lognorm) = paste0('rna_', colnames(single_rna_lognorm))
    colnames(single_adt_lognorm) = paste0('adt_', colnames(single_adt_lognorm))
    assay_list_indirect = list(
      RNA = single_rna_lognorm,    # dgCMatrix
      Multiome = as(mult_lognorm, 'dgCMatrix'), # dgCMatrix
      ADT = single_adt_lognorm  # dgCMatrix
    )

    lapply(assay_list_indirect, dim)

    # jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/CITE-mosaic-overlapFeature.jpg", 
    #   width = 4, height = 4, units = 'in', res = 300)
    mosaicDataUpSet(assay_list_indirect, plot = FALSE)
    # dev.off()

    # jpeg(file="/home/yxh/gitrepo/multi-omics-matching/StabMap-main/vignettes/CITE-mosaic-topology.jpg", 
    #   width = 4, height = 4, units = 'in', res = 300)
    mdt_indirect = mosaicDataTopology(assay_list_indirect)
    # plot(mdt_indirect)
    # dev.off()
    stab_indirect = stabMap(assay_list_indirect,
                            reference_list = c("Multiome"),
                            maxFeatures=5000,
                            plot = FALSE)

    # out_dir = paste0(out_dir, )
    write.csv(as.data.frame(stab_indirect), 
      file=paste0(out_dir, glue('p={p}_r={r}_embed.csv')),
      quote=F, row.names=T)
  }
}
