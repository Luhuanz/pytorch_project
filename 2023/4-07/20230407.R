library(igraph)

# 导入边和节点
setwd("F:/ggplot案例/R_example-master/2020.07.13Network_igraph")
edges <- read.table('edge.csv', header=T, sep=',') #导入边数据，里面可以包含每个边的频次数据或权重
vertices <- read.table('vertices.csv', header=T, sep=',') #导入节点数据，可以包含属性数据，如分类
edges ;vertices
#2)导入数据后，要转化成图数据才能用R作图，不同数据格式用不同方式=======
graph <- graph_from_data_frame(edges, directed = F, vertices=vertices) #directed = TRUE表示有方向,如果不需要点数据，可以设置vertices=NULL

#生成方式1（没有颜色分类）：======
igraph.options(vertex.size=3, vertex.label=NA, edge.arrow.size=0.5)

V(graph)$color <- colrs[V(graph)$color]
plot(graph,  
     layout=layout.reingold.tilford(graph,circular=T),  #layout.fruchterman.reingold表示弹簧式发散的布局，
     #其他还有环形布局layout.circle，分层布局layout.reingold.tilford，中心向外发散layout.reingold.tilford(graph,circular=T) ，核心布局layout_as_star，大型网络可视化layout_with_drl
     vertex.size=5,     #节点大小  
     vertex.shape='circle',    #节点不带边框none,,圆形边框circle,方块形rectangle  
     vertex.color="lightgreen",#设置颜色，其他如red,blue,cyan,yellow等
     vertex.label=vertices$name, #NULL表示不设置，为默认状态，即默认显示数据中点的名称，可以是中文。如果是NA则表示不显示任何点信息	 
     vertex.label.cex=0.8,    #节点字体大小  
     vertex.label.color='black',  #节点字体颜色,red  
     vertex.label.dist=0.4,   #标签和节点位置错开
     edge.arrow.size=0,#连线的箭头的大小,若为0即为无向图，当然有些数据格式不支持有向图  
     edge.width = 0.5, #连接线宽度
     edge.label=FALSE, #不显示连接线标签，默认为频次
     edge.color="gray")  #连线颜色 

l = layout.reingold.tilford(graph,circular=T)

#具体修改过程
V(graph)$size <- 8  #节点大小与点中心度成正比，中心度即与该点相连的点的总数
colrs <- c('#0096ff', "lightblue", "azure3","firebrick1")
V(graph)$color <- colrs[vertices$color] #根据类型设置颜色,按照类型分组
V(graph)$label.color <- 'black' #设置节点标记的颜色
V(graph)$label <- V(graph)$name 
#E(graph)$width <- E(graph)$fre #根据频次列设置边宽度
#E(graph)$label <- E(graph)$fre #根据频次列设置边标签
E(graph)$arrow.size=0.3 #设置箭头大小
#生成图
plot(graph, layout=l)
