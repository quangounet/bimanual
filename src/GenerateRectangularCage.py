def GenerateRectangularCage(lx=0.4, ly=0.5, lz=1.0, w=0.04, rho=500):
    """
    GenerateRectangularCage generates a rectagular cage with dimension
    lx x ly x lz. The cross section of the frame is w x w.
    """
    hw = 0.5*w
    hlx = 0.5*lx
    hly = 0.5*ly
    hlz = 0.5*lz

    bodyname = 'rectangular_cage_{0}x{1}x{2}'.\
    format(int(lx*100), int(ly*100), int(lz*100))
    xmlfilename = bodyname + '.kinbody.xml'

    XMLData = '<kinbody name="cage">'

    part01 = \
    """
      <body name="part01" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.0 0.0 1.0</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hw, hw, hlz, 0, 0, 0, rho)

    part02 = \
    """
      <body name="part02" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.847 0.686 0.439</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hw, hw, hlz, lx - w, 0, 0, rho)

    part03 = \
    """
      <body name="part03" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.847 0.686 0.439</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hw, hw, hlz, lx - w, ly - w, 0, rho)

    part04 = \
    """
      <body name="part04" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.847 0.686 0.439</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hw, hw, hlz, 0, ly - w, 0, rho)

    part05 = \
    """
      <body name="part05" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>1.0 0.0 0.0</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hlx - w, hw, hw, hlx - hw, 0, -hlz + hw, rho)


    part06 = \
    """
      <body name="part06" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.847 0.686 0.439</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hw, hly - w, hw, lx - w, hly - hw, -hlz + hw, rho)

    part07 = \
    """
      <body name="part07" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.847 0.686 0.439</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hlx - w, hw, hw, hlx - hw, ly - w, -hlz + hw, rho)


    part08 = \
    """
      <body name="part08" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.0 1.0 0.0</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hw, hly - w, hw, 0, hly - hw, -hlz + hw, rho)

    part09 = \
    """
      <body name="part09" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.847 0.686 0.439</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hlx - w, hw, hw, hlx - hw, 0, hlz - hw, rho)


    part10 = \
    """
      <body name="part10" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.847 0.686 0.439</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hw, hly - w, hw, lx - w, hly - hw, hlz - hw, rho)

    part11 = \
    """
      <body name="part11" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.847 0.686 0.439</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hlx - w, hw, hw, hlx - hw, ly - w, hlz - hw, rho)


    part12 = \
    """
      <body name="part12" type="dynamic">
        <geom type="box">
          <extents>{0} {1} {2}</extents>
          <diffusecolor>0.847 0.686 0.439</diffusecolor>
          <ambientcolor>0.6 0.6 0.6</ambientcolor>
        </geom>
        <translation>{3} {4} {5}</translation>
        <mass type="mimicgeom">
          <density>{6}</density>
        </mass>
      </body>
    """.format(hw, hly - w, hw, 0, hly - hw, hlz - hw, rho)
    
    XMLData += part01
    XMLData += part02
    XMLData += part03
    XMLData += part04
    XMLData += part05
    XMLData += part06
    XMLData += part07
    XMLData += part08
    XMLData += part09
    XMLData += part10
    XMLData += part11
    XMLData += part12

    XMLData += '</kinbody>'
    return XMLData
    # with open(xmlfilename, 'w') as f:
    #     f.write(XMLData)
