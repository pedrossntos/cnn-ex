import graphviz

def draw_neural_network():
    dot = graphviz.Digraph(format='png', engine='dot')
    
    dot.attr(rankdir='LR')
    
    layers = [
        ('Input Layer', '28x28x1'),
        ('Conv2D Layer 1', '32 filters 5x5'),
        ('Conv2D Layer 2', '64 filters 5x5'),
        ('MaxPooling Layer', '2x2 pool'),
        ('Dropout Layer 1', '0.25'),
        ('Flatten', 'Flattened output'),
        ('Dense Layer 1', '128 neurons'),
        ('Dropout Layer 2', '0.5'),
        ('Output Layer', '10 neurons')
    ]
    
    for i, (layer_name, layer_desc) in enumerate(layers):
        dot.node(f'Layer{i}', label=f'{layer_name}\n({layer_desc})', shape='rect', style='filled', fillcolor='lightblue', width='2.0', height='1.0', fontsize='12', fontname="Helvetica")
    
    for i in range(len(layers) - 1):
        dot.edge(f'Layer{i}', f'Layer{i+1}', dir='forward')
    
    return dot

if __name__ == "__main__":
    dot = draw_neural_network()
    dot.render("neural_network_architecture", cleanup=False)
    dot.view()
