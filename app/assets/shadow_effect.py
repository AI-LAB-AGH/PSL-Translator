from PyQt5.QtWidgets import QGraphicsDropShadowEffect

def shadow_effect():
    shadow_effect = QGraphicsDropShadowEffect()
    shadow_effect.setBlurRadius(5)
    shadow_effect.setXOffset(2)
    shadow_effect.setYOffset(2)
    
    return shadow_effect

def bottom_shadow_effect():
    shadow_effect = QGraphicsDropShadowEffect()
    shadow_effect.setBlurRadius(5)
    shadow_effect.setXOffset(0)
    shadow_effect.setYOffset(2)
    
    return shadow_effect
