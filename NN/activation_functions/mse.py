class MSE():
    # comments
    def activate(self, output, target):
        return (output - target) ** 2;
    
    def derivative(self, output, target):
        return 0;