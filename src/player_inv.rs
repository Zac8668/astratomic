use crate::prelude::*;
use bevy::render::{render_asset::RenderAssetUsages, render_resource::*};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Item {
    Atom(Atom),
    SmartTool,
}

impl Item {
    pub fn same(&self, item: Item) -> bool {
        if let Item::Atom(atom) = self {
            if let Item::Atom(atom2) = item {
                return atom.id == atom2.id;
            }
        }
        *self == item
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Slot {
    pub item: Item,
    // If none it means item is not stackable
    pub number: Option<u16>,
}

impl Slot {
    pub fn atom_full(id: u8, materials: &Materials) -> Self {
        Self {
            item: Item::Atom(Atom::new(id, materials)),
            number: Some(100),
        }
    }

    pub fn smart_tool() -> Self {
        Self {
            item: Item::SmartTool,
            number: None,
        }
    }
}

#[derive(Resource, Default, Debug)]
pub struct Inventory {
    pub slots: [Option<Slot>; 32],
    pub selected: usize,
    slot_ents: [Option<Entity>; 32],
    showing: bool,
}

impl Inventory {
    //Returns the index of the new item if it wasn't on the inventory
    fn add(&mut self, item: Item) -> Option<usize> {
        //Search if we have the item on inventory
        for slot in self.slots.iter_mut().flatten() {
            if slot.item.same(item) {
                if let Some(number) = &mut slot.number {
                    if *number < u16::MAX {
                        *number += 1;
                        return None;
                    }
                }
            }
        }
        //If we don't, search the first empty slot
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(Slot {
                    item,
                    number: Some(1),
                });
                return Some(i);
            }
        }
        None
    }

    pub fn can_add(&self, item: Item) -> bool {
        for slot in self.slots {
            if let Some(slot) = slot {
                if slot.item.same(item) {
                    if let Some(number) = slot.number {
                        if number < u16::MAX {
                            return true;
                        }
                    }
                }
            } else {
                return true;
            }
        }
        false
    }

    pub fn new(
        slots: [Option<Slot>; 32],
        slot_ents: [Option<Entity>; 32],
        selected: usize,
    ) -> Self {
        Self {
            slots,
            selected,
            slot_ents,
            showing: false,
        }
    }

    fn remove_one(&mut self, index: usize) -> Option<usize> {
        if let Some(slot) = self.slots[index] {
            if slot.number.is_none() || slot.number.unwrap() <= 1 {
                self.slots[index] = None;
                return Some(index);
            } else if let Some(slot) = &mut self.slots[index] {
                slot.number = Some(slot.number.unwrap() - 1);
            }
        }
        None
    }

    fn remove_one_selected(&mut self) -> Option<usize> {
        self.remove_one(self.selected)
    }

    fn _full(&self) -> bool {
        for slot in self.slots {
            if slot.is_none() {
                return false;
            }
        }
        true
    }
}

impl Drop for Inventory {
    fn drop(&mut self) {
        let file = File::create("assets/world/inventory").unwrap();
        let mut buffered = BufWriter::new(file);
        bincode::serialize_into(&mut buffered, &(&self.slots, &self.selected)).unwrap();
    }
}

#[derive(Component)]
pub struct SlotUi(usize);

#[derive(Component)]
pub struct ImageSlotUi(usize, Entity);

#[derive(Component)]
pub struct InvUi;

pub fn inv_setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
    mut images: ResMut<Assets<Image>>,
    mut inv: ResMut<Inventory>,
) {
    let materials = materials.0.get(&materials.1 .0).unwrap();

    let (mut slots, selected): ([Option<Slot>; 32], usize);
    if let Ok(file) = File::open("assets/world/inventory") {
        let mut buffered = BufReader::new(file);
        (slots, selected) = bincode::deserialize_from(&mut buffered).unwrap();
    } else {
        (slots, selected) = ([None; 32], 0);
        slots[0] = Some(Slot::smart_tool());
        slots[1] = Some(Slot::atom_full(2, materials));
        slots[2] = Some(Slot::atom_full(3, materials));
        slots[3] = Some(Slot::atom_full(4, materials));
        slots[4] = Some(Slot::atom_full(5, materials));
        slots[5] = Some(Slot::atom_full(9, materials));

        let file = File::create("assets/world/inventory").unwrap();
        let mut buffered = BufWriter::new(file);
        bincode::serialize_into(&mut buffered, &(slots, selected)).unwrap();
    }

    let mut slot_ents = [None; 32];

    commands
        .spawn((Node {
            justify_self: JustifySelf::Center,
            display: Display::Grid,
            grid_template_columns: RepeatedGridTrack::flex(8, 1.0),
            grid_template_rows: RepeatedGridTrack::flex(4, 1.0),
            row_gap: Val::Px(35.),
            column_gap: Val::Px(35.),
            ..Default::default()
        },))
        .with_children(|parent| {
            for i in 0..32 {
                let mut parent = parent.spawn((
                    Node {
                        width: Val::Px(54.),
                        height: Val::Px(54.),
                        margin: UiRect::all(Val::Px(20.)),
                        align_items: AlignItems::Center,
                        //justify_content: JustifyContent::Center,
                        justify_self: JustifySelf::Center,
                        border: UiRect::all(Val::Percent(12.)),
                        ..default()
                    },
                    BackgroundColor(Color::NONE),
                    BorderColor(Color::NONE),
                    Outline {
                        width: Val::Px(6.),
                        offset: Val::Px(6.),
                        color: Color::WHITE,
                    },
                    SlotUi(i),
                    Button,
                ));
                parent.with_child(());

                if let Some(slot) = slots[i] {
                    let image = match slot.item {
                        Item::Atom(atom) => {
                            let size: u32 = 8;
                            let mut data = vec![];
                            for _ in 0..(size.pow(2)) {
                                let color = Atom::new(atom.id, materials).color.to_vec();
                                for u in color {
                                    data.push(u);
                                }
                            }

                            images.add(Image::new(
                                Extent3d {
                                    height: size,
                                    width: size,
                                    ..Default::default()
                                },
                                TextureDimension::D2,
                                data,
                                TextureFormat::Rgba8UnormSrgb,
                                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
                            ))
                        }
                        Item::SmartTool => asset_server.load("player/player_tool.png"),
                    };

                    let image_node_style = match slot.item {
                        // Procedural icons are square; fill the slot.
                        Item::Atom(_) => Node {
                            width: Val::Percent(100.),
                            height: Val::Percent(100.),
                            ..Default::default()
                        },
                        // Keep the PNG aspect ratio (8x6 => 4/3).
                        Item::SmartTool => Node {
                            width: Val::Percent(100.),
                            height: Val::Auto,
                            aspect_ratio: Some(4.0 / 3.0),
                            ..Default::default()
                        },
                    };

                    parent.with_child((
                        ImageNode {
                            image,
                            ..Default::default()
                        },
                        image_node_style,
                        ImageSlotUi(i, parent.id()),
                    ));
                }

                if i >= 8 {
                    parent.insert((InvUi, Visibility::Hidden));
                }
                slot_ents[i] = Some(parent.id());
            }
        });

    *inv = Inventory::new(slots, slot_ents, selected);
}

#[derive(Component)]
pub struct NumberUi(pub Entity);

pub fn spawn_numbers(
    mut commands: Commands,
    quaver_font: Res<QuaverFont>,
    slots: Query<(&ImageSlotUi, Entity)>,
    inventory: Res<Inventory>,
) {
    let quaver_font: Handle<Font> = quaver_font.0.clone();
    let text_style = TextFont {
        font: quaver_font,
        font_size: 11.,
        ..Default::default()
    };
    let text_color = TextColor(Color::srgb(0.9, 0.9, 0.9));

    for (index, ent) in slots.iter() {
        if let Some(slot) = inventory.slots[index.0] {
            if let Some(number) = slot.number {
                let number_ent = commands
                    .spawn((
                        Text::new(format!("{number}")),
                        text_style.clone(),
                        text_color,
                        Node {
                            position_type: PositionType::Absolute,
                            right: Val::Px(2.0),
                            bottom: Val::Px(2.0),
                            ..default()
                        },
                        NumberUi(ent),
                    ))
                    .id();

                commands.entity(ent).add_child(number_ent);

                if index.0 >= 8 {
                    commands.entity(number_ent).insert(Visibility::Hidden);
                }
            }
        }
    }
}

pub fn update_numbers(
    mut commands: Commands,
    slots: Query<&ImageSlotUi, Without<NumberUi>>,
    mut numbers: Query<(&NumberUi, &mut Text, Entity, &mut Visibility), Without<ImageSlotUi>>,
    inventory: Res<Inventory>,
) {
    for (number_ui, mut text, ent, mut visibility) in numbers.iter_mut() {
        if let Ok(slot) = slots.get(number_ui.0) {
            if slot.0 >= 8 && !inventory.showing {
                *visibility = Visibility::Hidden;
            } else if slot.0 < 8 || inventory.showing {
                *visibility = Visibility::Visible;
            }

            if let Some(slot) = inventory.slots[slot.0] {
                if let Some(n) = slot.number {
                    text.0 = n.to_string();
                } else {
                    commands.entity(ent).despawn_recursive();
                }
            } else {
                commands.entity(ent).despawn_recursive();
            }
        } else {
            commands.entity(ent).despawn_recursive();
        }
    }
}

#[derive(Event)]
pub enum ItemEvent {
    AddOne(Item),
    RemoveSelected,
    //First is from, second is to
    MoveToEmpty(usize, usize),
}

pub fn item_events(
    mut commands: Commands,
    mut ev_items: EventReader<ItemEvent>,
    mut inv: ResMut<Inventory>,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
    quaver_font: Res<QuaverFont>,
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
) {
    let materials = materials.0.get(&materials.1 .0).unwrap();
    let quaver_font: Handle<Font> = quaver_font.0.clone();

    for ev in ev_items.read() {
        match ev {
            ItemEvent::AddOne(item) => {
                if let Some(index) = inv.add(*item) {
                    let ent = inv.slot_ents[index].unwrap();
                    let image = match item {
                        Item::Atom(atom) => {
                            let size: u32 = 8;
                            let mut data = vec![];
                            for _ in 0..(size.pow(2)) {
                                let color = Atom::new(atom.id, materials).color.to_vec();
                                for u in color {
                                    data.push(u);
                                }
                            }

                            images.add(Image::new(
                                Extent3d {
                                    height: size,
                                    width: size,
                                    ..Default::default()
                                },
                                TextureDimension::D2,
                                data,
                                TextureFormat::Rgba8UnormSrgb,
                                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
                            ))
                        }
                        Item::SmartTool => asset_server.load("player/player_tool.png"),
                    };

                    let image_node_style = match item {
                        Item::Atom(_) => Node {
                            width: Val::Percent(100.),
                            height: Val::Percent(100.),
                            ..Default::default()
                        },
                        Item::SmartTool => Node {
                            width: Val::Percent(100.),
                            height: Val::Auto,
                            aspect_ratio: Some(4.0 / 3.0),
                            ..Default::default()
                        },
                    };

                    let image_ent = commands
                        .spawn((
                            ImageNode {
                                image,
                                ..Default::default()
                            },
                            image_node_style,
                            ImageSlotUi(index, ent),
                        ))
                        .id();

                    commands.entity(ent).add_child(image_ent);

                    let slot = inv.slots[index].unwrap();
                    if let Some(number) = slot.number {
                        let text_style = TextFont {
                            font: quaver_font.clone(),
                            font_size: 11.,
                            ..Default::default()
                        };
                        let text_color = TextColor(Color::srgb(0.9, 0.9, 0.9));

                        let number_ent = commands
                            .spawn((
                                Text::new(format!("{number}")),
                                text_style.clone(),
                                text_color,
                                Node {
                                    position_type: PositionType::Absolute,
                                    right: Val::Px(2.0),
                                    bottom: Val::Px(2.0),
                                    ..default()
                                },
                                NumberUi(image_ent),
                            ))
                            .id();

                        commands.entity(image_ent).add_child(number_ent);

                        if !inv.showing && index >= 8 {
                            commands.entity(number_ent).insert(Visibility::Hidden);
                        } else if inv.showing || index < 8 {
                            commands.entity(number_ent).insert(Visibility::Visible);
                        }
                    }
                }
            }
            ItemEvent::RemoveSelected => {
                if let Some(index) = inv.remove_one_selected() {
                    if let Some(ent) = inv.slot_ents[index] {
                        commands.entity(ent).despawn_descendants();
                    }
                }
            }
            ItemEvent::MoveToEmpty(from, to) => {
                let slot = inv.slots[*from].unwrap();
                inv.slots[*to] = Some(slot);
                if from != to {
                    inv.slots[*from] = None;
                }
            }
        }
    }
}

pub fn show_inventory(
    inputs: Res<Inputs>,
    mut inventory: ResMut<Inventory>,
    mut inv_ui: Query<&mut Visibility, With<InvUi>>,
    mut slots: Query<&mut Outline, With<SlotUi>>,
) {
    //Change Selected Outline
    let mut selected_outline = slots
        .get_mut(inventory.slot_ents[inventory.selected].unwrap())
        .unwrap();

    *selected_outline = Outline {
        color: Color::WHITE,
        width: Val::Px(9.),
        offset: Val::Px(6.),
    };

    //Toggle inventory open
    if !inventory.showing && inputs.inventory_toggle {
        inventory.showing = true;
        for mut vis in inv_ui.iter_mut() {
            *vis = Visibility::Visible;
        }
    } else if inputs.inventory_toggle && inventory.showing {
        inventory.showing = false;
        for mut vis in inv_ui.iter_mut() {
            *vis = Visibility::Hidden;
        }
    }
}

fn clear_selected(mut slots: Query<&mut Outline, With<SlotUi>>) {
    for mut outline in slots.iter_mut() {
        *outline = Outline {
            color: Color::linear_rgb(0.4, 0.4, 0.4),
            width: Val::Px(6.),
            offset: Val::Px(6.),
        };
    }
}

fn slot_button(
    mut commands: Commands,
    slots: Query<
        (&Interaction, &Children, &SlotUi, Entity),
        (Changed<Interaction>, (With<Button>, Without<FollowMouse>)),
    >,
    mut slot_images: Query<(&mut Node, &FollowMouse, Entity, &mut ImageSlotUi)>,
    slot_images_nf: Query<&ImageSlotUi, Without<FollowMouse>>,
    window: Query<&Window>,
    mut ev_item: EventWriter<ItemEvent>,
    inventory: Res<Inventory>,
) {
    let Some(cursor_position) = window.get_single().unwrap().cursor_position() else {
        return;
    };

    for (mut node, _, _, _) in slot_images.iter_mut() {
        node.left = Val::Px(cursor_position.x - 50. / 2.);
        node.top = Val::Px(cursor_position.y - 50. / 2.);
    }

    if let Ok((mut node, _, ent, im_slot_ui)) = slot_images.get_single_mut() {
        if !inventory.showing {
            commands.entity(ent).remove::<FollowMouse>();
            commands.entity(im_slot_ui.1).add_child(ent);
            node.width = Val::Percent(100.);
            node.height = Val::DEFAULT;
            node.left = Val::ZERO;
            node.top = Val::ZERO;
            node.position_type = PositionType::Relative;
        }
    }

    for (interaction, children, slot_ui, slot_ent) in slots.iter() {
        if *interaction == Interaction::Pressed {
            if let Ok((mut node, _, ent, mut im_slot_ui)) = slot_images.get_single_mut() {
                if slot_ui.0 == im_slot_ui.0 || inventory.slots[slot_ui.0].is_none() {
                    commands.entity(ent).remove::<FollowMouse>();
                    commands.entity(slot_ent).add_child(ent);
                    node.width = Val::Percent(100.);
                    node.height = Val::DEFAULT;
                    node.left = Val::ZERO;
                    node.top = Val::ZERO;
                    node.position_type = PositionType::Relative;

                    ev_item.send(ItemEvent::MoveToEmpty(im_slot_ui.0, slot_ui.0));
                    im_slot_ui.0 = slot_ui.0;
                    im_slot_ui.1 = slot_ent;
                }
            } else if let Some(ent) = children.last() {
                if inventory.showing && slot_images_nf.get(*ent).is_ok() {
                    commands.entity(slot_ent).remove_children(&[*ent]);

                    commands.entity(*ent).insert((
                        FollowMouse,
                        Node {
                            width: Val::Px(43.),
                            position_type: PositionType::Absolute,
                            left: Val::Px(cursor_position.x - 50. / 2.),
                            top: Val::Px(cursor_position.y - 50. / 2.),
                            ..default()
                        },
                    ));
                }
            }
        }
    }
}

#[derive(Component)]
pub struct FollowMouse;

pub struct PlayerInvPlugin;
impl Plugin for PlayerInvPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            OnEnter(GameState::Game),
            (
                inv_setup.after(manager_setup),
                spawn_numbers.after(inv_setup),
            ),
        )
        .add_event::<ItemEvent>()
        .init_resource::<Inventory>()
        .add_systems(
            Update,
            (
                update_numbers.after(item_events).after(show_inventory),
                item_events.after(show_inventory),
                show_inventory,
                slot_button.before(update_numbers).before(item_events),
            )
                .run_if(in_state(GameState::Game)),
        )
        .add_systems(PreUpdate, clear_selected.run_if(in_state(GameState::Game)));
    }
}
